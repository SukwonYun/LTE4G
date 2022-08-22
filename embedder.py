import scipy.sparse as sp
import numpy as np
import torch
import data_load
import utils
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

class embedder:
    def __init__(self, args):
        if args.gpu == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

        # Load data - Cora, CiteSeer, cora_full
        self._dataset = data_load.Dataset(root="data", dataset=args.dataset, is_normalize=args.is_normalize, add_self_loop=args.add_sl)
        self.edge_index = self._dataset.edge_index
        adj = self._dataset.adj
        features = self._dataset.features
        labels = self._dataset.labels
        class_sample_num = 20
        im_class_num = args.im_class_num

        # Natural Setting
        if args.im_ratio == 1:
            args.criterion = 'mean'
            labels, og_to_new = utils.refine_label_order(labels)
            idx_train, idx_val, idx_test, class_num_mat = utils.split_natural(labels, og_to_new)
            samples_per_label = torch.tensor(class_num_mat[:,0])

        # Manual Setting
        elif args.im_ratio in [0.2, 0.1, 0.05]:
            args.criterion = 'max'
            labels, og_to_new = utils.refine_label_order(labels)
            c_train_num = []
            for i in range(labels.max().item() + 1):
                if i > labels.max().item() - im_class_num:  # last classes belong to minority classes
                    c_train_num.append(int(class_sample_num * args.im_ratio))
                else:
                    c_train_num.append(class_sample_num)

            idx_train, idx_val, idx_test, class_num_mat = utils.split_manual(labels, c_train_num, og_to_new)
            samples_per_label = torch.tensor(class_num_mat[:,0])
        
        # LT version
        elif (args.im_ratio == 0.01) & (args.dataset in ['Cora', 'CiteSeer']):
                args.criterion = 'mean'
                data = self._dataset
                labels = data.labels
                n_cls = labels.max().item() + 1
                data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
                ## Data statistic ##
                stats = labels[data_train_mask]
                n_data = []
                for i in range(n_cls):
                    data_num = (stats == i).sum()
                    n_data.append(int(data_num.item()))

                class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = utils.make_longtailed_data_remove(data.edge_index,
                                                                        labels, n_data, n_cls, 100, data_train_mask.clone())

                edge = data.edge_index[:,train_edge_mask]              
                if args.add_sl:
                    edge = remove_self_loops(edge)[0]
                    edge = add_self_loops(edge)[0]
                adj = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0,:], edge[1,:])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
                adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

                labels, og_to_new = utils.refine_label_order(labels)

                if args.dataset=='Cora': # change label 1<->2 in order to maintain descending order
                    labels[labels==1] = 100
                    labels[labels==2] = 1
                    labels[labels==100] = 2

                elif args.dataset=='CiteSeer':  # change label 0<->1 in order to maintain descending order
                    labels[labels==0] = 100
                    labels[labels==1] = 0
                    labels[labels==100] = 1

                total_nodes = len(labels)                
                idx_train = torch.tensor(range(total_nodes))[data_train_mask]
                idx_val = torch.tensor(range(total_nodes))[data_val_mask]
                idx_test = torch.tensor(range(total_nodes))[data_test_mask]

                idx_train, idx_val, idx_test, class_num_mat = utils.split_manual_lt(labels, idx_train, idx_val, idx_test)
                samples_per_label = torch.tensor(class_num_mat[:,0])

        # Set embeder
        if 'lte4g' in args.embedder:
            manual = True if args.im_ratio in [0.2, 0.1, 0.05] else False
            idx_train_set_class, ht_dict_class = utils.separate_ht(samples_per_label, labels, idx_train, method=args.sep_class, manual=manual)
            idx_train_set, degree_dict, degrees, above_head, below_tail  = utils.separate_class_degree(adj, idx_train_set_class, below=args.sep_degree)
            
            idx_val_set = utils.separate_eval(idx_val, labels, ht_dict_class, degrees, above_head, below_tail)
            idx_test_set = utils.separate_eval(idx_test, labels, ht_dict_class, degrees, above_head, below_tail)

            args.sep_point = len(ht_dict_class['H'])

            self.idx_train_set_class = idx_train_set_class
            self.degrees = degrees
            self.above_head = above_head
            self.below_tail = below_tail

            print('Above Head Degree:', above_head)
            print('Below Tail Degree:', below_tail)
            
            self.idx_train_set = {}
            self.idx_val_set = {}
            self.idx_test_set = {}
            for sep in ['HH', 'HT', 'TH', 'TT']:
                self.idx_train_set[sep] = idx_train_set[sep].to(args.device)
                self.idx_val_set[sep] = idx_val_set[sep].to(args.device)
                self.idx_test_set[sep] = idx_test_set[sep].to(args.device)

        adj = utils.normalize_adj(adj) if args.adj_norm_1 else utils.normalize_sym(adj)

        self.adj = adj.to(args.device)
        self.features = features.to(args.device)
        self.labels = labels.to(args.device)
        self.class_sample_num = class_sample_num
        self.im_class_num = im_class_num

        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.samples_per_label = samples_per_label
        self.class_num_mat = class_num_mat
        print(class_num_mat)

        args.nfeat = features.shape[1]
        args.nclass = labels.max().item() + 1
        args.im_class_num = im_class_num

        self.args = args
