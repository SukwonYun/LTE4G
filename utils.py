import numpy as np
import torch
import random
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import os.path as osp
import os
import logging
import sys
from torch_scatter import scatter_add

# LT dataset from GraphENS
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    # Check whether inv_indices is correct
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        # Check whether the number of class is greater than or equal to 1
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)

    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]
    # print(class_num_list);input()

    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) * original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = (row_mask * col_mask).type(torch.bool)

            # Compute degree
            degree = scatter_add(torch.ones_like(row[edge_mask]), row[edge_mask]).to(row.device)
            if len(degree) < len(label):
                degree = torch.cat([degree, degree.new_zeros(len(label)-len(degree))], dim=0)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = (row_mask * col_mask).type(torch.bool)

    train_mask = (node_mask * train_mask).type(torch.bool)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) * train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask

def split_manual(labels, c_train_num, idx_map):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25 
    c_num_mat[:,2] = 55 

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]: c_train_num[i]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_train_num[i]+c_num_mat[i,1]: c_train_num[i]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def split_manual_lt(labels, idx_train, idx_val, idx_test):
    num_classes = len(set(labels.tolist()))
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels[idx_train]==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        val_lists = list(map(int,idx_val[labels[idx_val]==i]))
        test_lists = list(map(int,idx_test[labels[idx_test]==i]))
        random.shuffle(val_lists)
        random.shuffle(test_lists)

        c_num_mat[i,0] = len(c_idx)

        val_idx = val_idx + val_lists[:c_num_mat[i,1]]
        test_idx = test_idx + test_lists[:c_num_mat[i,2]]

    train_idx = torch.LongTensor(idx_train)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def split_natural(labels, idx_map):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        idx = list(idx_map.keys())[list(idx_map.values()).index(i)]
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('OG:{:d} -> NEW:{:d}-th class sample number: {:d}'.format(idx, i, len(c_idx)))
        c_num = len(c_idx)

        if c_num == 3:
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
        else:
            random.shuffle(c_idx)
            c_idxs.append(c_idx)
            c_num_mat[i,0] = int(c_num*0.1) # 10% for train
            c_num_mat[i,1] = int(c_num*0.1) # 10% for validation
            c_num_mat[i,2] = int(c_num*0.8) # 80% for test
        # print('[{}-th class] Total: {} | Train: {} | Val: {} | Test: {}'.format(i,len(c_idx), c_num_mat[i,0], c_num_mat[i,1], c_num_mat[i,2]))

        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat

def separate_class_degree(adj, idx_train_set_class, above_head=None, below_tail=None, below=None, rand=False, is_eval=False):
    idx_train_set = {}
    idx_train_set['HH'] = []
    idx_train_set['HT'] = []
    idx_train_set['TH'] = []
    idx_train_set['TT'] = []

    adj_dense = adj.to_dense()
    adj_dense[adj_dense != 0] = 1
    degrees = np.array(list(map(int, torch.sum(adj_dense, dim=0))))

    if rand:
        for sep in ['H', 'T']:
            idxs = np.array(idx_train_set_class[sep])
            np.random.shuffle(idxs)
        
            idx_train_set[sep+'H'] = idxs[:int(len(idxs)/2)]
            idx_train_set[sep+'T'] = idxs[int(len(idxs)/2):]
            
        for idx in ['HH', 'HT', 'TH', 'TT']:
            random.shuffle(idx_train_set[idx])
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

            degree_dict = {}
            above_head = 0
            below_tail = 0
        
        return idx_train_set, degree_dict, degrees, above_head, below_tail


    if not is_eval:
        above_head = {}
        below_tail = {}
        degree_dict = {}

        for sep in ['H', 'T']:
            if len(idx_train_set_class[sep]) == 0:
                continue

            elif len(idx_train_set_class[sep]) == 1:
                idx = idx_train_set_class[sep]
                if sep == 'H':
                    rand = random.choice(['HH', 'HT'])
                    idx_train_set[rand].append(int(idx))
                elif sep == 'T':
                    rand = random.choice(['TH', 'TT'])
                    idx_train_set[rand].append(int(idx))

            else:
                degrees_idx_train = degrees[idx_train_set_class[sep]]

                above_head = below + 1
                below_tail = below
                gap_head = abs(degrees_idx_train - (below+1))
                gap_tail = abs(degrees_idx_train - below)

                if sep == 'H':
                    idx_train_set['HH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                    idx_train_set['HT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                    if sum(gap_head == gap_tail) > 0:
                        for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                            rand = random.choice(['HH', 'HT'])
                            idx_train_set[rand].append(int(idx))

                elif sep == 'T':
                    idx_train_set['TH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                    idx_train_set['TT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                    if sum(gap_head == gap_tail) > 0:
                        for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                            rand = random.choice(['TH', 'TT'])
                            idx_train_set[rand].append(int(idx))

        for idx in ['HH', 'HT', 'TH', 'TT']:
            random.shuffle(idx_train_set[idx])
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

        return idx_train_set, degree_dict, degrees, above_head, below_tail
    
    elif is_eval:
        for sep in ['H', 'T']:
            if len(idx_train_set_class[sep]) == 0:
                continue

            else:
                degrees_idx_train = degrees[idx_train_set_class[sep]]

                gap_head = abs(degrees_idx_train - above_head)
                gap_tail = abs(degrees_idx_train - below_tail)

                if sep == 'H':
                    if len(idx_train_set_class[sep]) == 1:
                        if gap_head < gap_tail:
                            idx_train_set['HH'].append(int(idx_train_set_class[sep]))
                        elif gap_tail < gap_head:
                            idx_train_set['HT'].append((idx_train_set_class[sep]))
                        else:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['HH', 'HT'])
                                idx_train_set[rand].append(int(idx))
                    else:
                        idx_train_set['HH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                        idx_train_set['HT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                        if sum(gap_head == gap_tail) > 0:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['HH', 'HT'])
                                idx_train_set[rand].append(int(idx))

                elif sep == 'T':
                    if len(idx_train_set_class[sep]) == 1:
                        if gap_head < gap_tail:
                            idx_train_set['TH'].append(int(idx_train_set_class[sep]))
                        elif gap_tail < gap_head:
                            idx_train_set['TT'].append(int(idx_train_set_class[sep]))
                        else:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['TH', 'TT'])
                                idx_train_set[rand].append(int(idx))
                    else:
                        idx_train_set['TH'] = list(map(int,idx_train_set_class[sep][gap_head < gap_tail]))
                        idx_train_set['TT'] = list(map(int,idx_train_set_class[sep][gap_tail < gap_head]))

                        if sum(gap_head == gap_tail) > 0:
                            for idx in idx_train_set_class[sep][gap_head == gap_tail]:
                                rand = random.choice(['TH', 'TT'])
                                idx_train_set[rand].append(int(idx))
            
        for idx in ['HH', 'HT', 'TH', 'TT']:
            idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])
                
        return idx_train_set

def separate_eval(idx_eval, labels, ht_dict_class, degrees, above_head, below_tail):
    idx_eval_set = {}
    idx_eval_set['HH'] = []
    idx_eval_set['HT'] = []
    idx_eval_set['TH'] = []
    idx_eval_set['TT'] = []
    
    for idx in idx_eval:
        label = int(labels[idx])
        degree = int(degrees[idx])
        if (label in ht_dict_class['H']) and (degree >= above_head):
            idx_eval_set['HH'].append(int(idx))

        elif (label in ht_dict_class['H']) and (degree <= below_tail):
            idx_eval_set['HT'].append(int(idx))

        elif (label in ht_dict_class['T']) and (degree >= above_head):
            idx_eval_set['TH'].append(int(idx))

        elif (label in ht_dict_class['T']) and (degree <= below_tail):
            idx_eval_set['TT'].append(int(idx))
        
    
    for idx in ['HH', 'HT', 'TH', 'TT']:
        random.shuffle(idx_eval_set[idx])
        idx_eval_set[idx] = torch.LongTensor(idx_eval_set[idx])
            
    return idx_eval_set

def separate_ht(samples_per_label, labels, idx_train, method='pareto_28', rand=False, manual=False):
    class_dict = {}
    idx_train_set = {}

    if rand:
        ht_dict = {}
        arr = np.array(idx_train)
        np.random.shuffle(arr)
        sample_num = int(idx_train.shape[0]/2)
        sample_label_num = int(len(labels.unique())/2)
        label_list = np.array(labels.unique())
        np.random.shuffle(label_list)
        ht_dict['H'] = label_list[0:sample_label_num]
        ht_dict['T'] = label_list[sample_label_num:]

        idx_train_set['H'] = arr[0:sample_num]
        idx_train_set['T'] = arr[sample_num:]

    elif manual:
        ht_dict = {}
        samples = samples_per_label
        point = np.arange(len(samples_per_label)-1)[list(map(lambda x: samples[x] != samples[x+1], range(len(samples)-1)))][0]
        label_list = np.array(labels.unique())
        ht_dict['H'] = label_list[0:point+1]
        ht_dict['T'] = label_list[point+1:]

        print('Samples per label:', samples_per_label)
        print('Separation:', ht_dict.items())

        idx_train_set['H'] = []
        idx_train_set['T'] = []
        for label in label_list:
            idx = 'H' if label <= point else 'T'
            idx_train_set[idx].extend(torch.LongTensor(idx_train[labels[idx_train] == label]))
            
    else:
        ht_dict = separator_ht(samples_per_label, method) # H/T

        print('Samples per label:', samples_per_label)
        print('Separation:', ht_dict.items())

        for idx, value in ht_dict.items():
            class_dict[idx] = []
            idx_train_set[idx] = []
            idx = idx
            label_list = value

            for label in label_list:
                class_dict[idx].append(label)
                idx_train_set[idx].extend(torch.LongTensor(idx_train[labels[idx_train] == label]))
            
    for idx in list(ht_dict.keys()):
        random.shuffle(idx_train_set[idx])
        idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

    return idx_train_set, ht_dict


def separator_ht(dist, method='pareto_28', degree=False): # Head / Tail separator
    head = int(method[-2]) # 2 in pareto_28
    tail = int(method[-1]) # 8 in pareto_28
    head_idx = int(len(dist) * (head/10))
    ht_dict = {}

    if head_idx == 0:
        ht_dict['H'] = list(range(0, 1))
        ht_dict['T'] = list(range(1, len(dist)))
        return ht_dict

    else:
        crierion = dist[head_idx].item()

        case1_h = sum(np.array(dist) >= crierion)
        case1_t = sum(np.array(dist) < crierion)

        case2_h = sum(np.array(dist) > crierion)
        case2_t = sum(np.array(dist) <= crierion)

        gap_case1 = abs(case1_h/case1_t - head/tail)
        gap_case2 = abs(case2_h/case2_t - head/tail)

        if gap_case1 < gap_case2:
            idx = sum(np.array(dist) >= crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        elif gap_case1 > gap_case2:
            idx = sum(np.array(dist) > crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        else:
            rand = random.choice([1, 2])
            if rand == 1:
                idx = sum(np.array(dist) >= crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))
            else:
                idx = sum(np.array(dist) > crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))

        return ht_dict

def accuracy(output, labels, sep_point=None, sep=None, pre=None):
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds= output

    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def classification(output, labels, sep_point=None, sep=None):
    target_names = []
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        pred = output.max(1)[1].type_as(labels)
        for i in labels.unique():
            target_names.append(f'class_{int(i)}')

        return classification_report(labels, pred)

def confusion(output, labels, sep_point=None, sep=None):
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        
        pred = output.max(1)[1].type_as(labels)
    
        return confusion_matrix(labels, pred)

def performance_measure(output, labels, sep_point=None, sep=None, pre=None):
    acc = accuracy(output, labels, sep_point=sep_point, sep=sep, pre=pre)*100

    if len(labels) == 0:
        return np.nan
    
    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)
    
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point # [4,5,6] -> [0,1,2]

    macro_F = f1_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    gmean = geometric_mean_score(labels.cpu().detach(), output.cpu().detach(), average='macro')*100
    bacc = balanced_accuracy_score(labels.cpu().detach(), output.cpu().detach())*100

    return acc, macro_F, gmean, bacc

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    
    adj_tgt[adj_tgt != 0] = 1

    edge_num = adj_tgt.nonzero().shape[0] #number of non-zero
    total_num = adj_tgt.shape[0]**2 #possible edge

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2) # element-wise

    return loss

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def refine_label_order(labels):
    print('Refine label order, Many to Few')
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    sorted_index = np.argsort(num_labels_each_class)[::-1]
    idx_map = {sorted_index[i]:i for i in range(num_labels)}
    new_labels = np.vectorize(idx_map.get)(labels.numpy())

    return labels.new(new_labels), idx_map

def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))
    return sum_m 

def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()
    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    
    return adj

def normalize_sym(adj):
    """Symmetric-normalize sparse matrix"""
    deg = torch.sum(adj.to_dense(), dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_sqrt = torch.diag(deg_inv_sqrt).to_sparse()

    adj = torch.spmm(deg_inv_sqrt, adj.to_dense()).to_sparse()
    adj = torch.spmm(adj, deg_inv_sqrt.to_dense()).to_sparse()

    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def scheduler(epoch, curriculum_ep=500, func='convex'):
    if func == 'convex':
        return np.cos((epoch * np.pi) / (curriculum_ep * 2))
    elif func == 'concave':
        return np.power(0.99, epoch)
    elif func == 'linear':
        return 1 - (epoch / curriculum_ep)
    elif func == 'composite':
        return (1/2) * np.cos((epoch*np.pi) / curriculum_ep) + 1/2

def setupt_logger(save_dir, text, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    # for each in logger.handlers:
    #     logger.removeHandler(each)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")
    return logger

def set_filename(args):
    rec_with_ep_pre = 'True_ep_pre_' + str(args.ep_pre) + '_rw_' + str(args.rw) if args.rec else 'False'

    if args.im_ratio == 1: # Natural Setting
        results_path = f'./results/natural/{args.dataset}'
        logs_path = f'./logs/natural/{args.dataset}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/natural/{args.dataset}/({args.layer}){textname}', 'w')
        file = f'./logs/natural/{args.dataset}/({args.layer})lte4g.txt'
        
    else: # Manual Imbalance Setting (0.2, 0.1, 0.05)
        results_path = f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        logs_path = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer}){textname}', 'w')
        file = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer})lte4g.txt'
        
    return text, file