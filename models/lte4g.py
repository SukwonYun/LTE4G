from embedder import embedder
import torch.nn as nn
import layers
import torch.optim as optim
import utils
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from copy import deepcopy
import datetime
from tqdm import trange
import os

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

class lte4g():
    def __init__(self, args):
        self.args = args

    def training(self):
        text, filename = utils.set_filename(self.args)

        logger = utils.setupt_logger('./', '-', filename)
        seed_result = {}

        seed_result['acc'] = []
        seed_result['macro_F'] = []
        seed_result['gmeans'] = []
        seed_result['bacc'] = []
        
        for seed in range(5, 5+self.args.num_seed):
            print(f'============== seed:{seed} ==============')
            utils.seed_everything(seed)
            print('seed:', seed, file=text)
            self = embedder(self.args)
            print('Samples per label:', list(self.class_num_mat[:,0]), file=text)

            print('-----Number of training samples in each Expert-----')
            print('HH:', len(self.idx_train_set['HH']))
            print('HT:', len(self.idx_train_set['HT']))
            print('TH:', len(self.idx_train_set['TH']))
            print('TT:', len(self.idx_train_set['TT']))
            
            print('HH:', len(self.idx_train_set['HH']),file=text)
            print('HT:', len(self.idx_train_set['HT']),file=text)
            print('TH:', len(self.idx_train_set['TH']),file=text)
            print('TT:', len(self.idx_train_set['TT']),file=text)
            print()

            print(file=text)

            model = modeler(self.args, self.adj).to(self.args.device)
            self.degrees = torch.tensor(self.degrees).to(self.args.device)
            avg_degree = []
            for i, label in enumerate(self.labels.unique()):
                avg_degree.append((sum(self.degrees[self.idx_train][self.labels[self.idx_train] == label]) / sum(self.labels[self.idx_train] == label)).item())
            
            avg_degree = torch.tensor(avg_degree).to(self.args.device)
            class_weight = 1 / torch.tensor(self.class_num_mat[:,0]).to(self.args.device)

            # ground truch of head / tail separation
            data_set = [self.idx_train, self.idx_val, self.idx_test]
            ht_gt = {}
            for data in data_set:
                ht_gt[data] = (self.labels[data] >= self.args.sep_point).type(torch.long) # head to '0', tail to '1'
        
            # ======================================= Encoder Training ======================================= #
            if self.args.pretrained_encoder:
                ####### Load Pre-trained Original Imbalance graph #######
                print('Load Pre-trained Encoder')       
                rec_with_ep_pre = 'True_ep_pre_' + str(self.args.ep_pre) + '_rw_' + str(self.args.rw) if self.args.rec else 'False'
                encoder_info = f'cls_{self.args.cls_og}_cw_{self.args.class_weight}_gamma_{self.args.gamma}_alpha_{self.args.alpha}_lr_{self.args.lr}_dropout_{self.args.dropout}_rec_{rec_with_ep_pre}_seed_{seed}.pkl'
                if self.args.im_ratio != 1: # manual
                    pretrained_encoder = torch.load(f'./pretrained/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}/'+encoder_info)
                else: # natural
                    pretrained_encoder = torch.load(f'./pretrained/natural/{self.args.dataset}/'+encoder_info)

                model.load_state_dict(pretrained_encoder.state_dict())

            else:
                ####### Pre-train Original Imbalance graph #######
                print('Start Pre-training Encoder')       
                best_encoder = None
                optimizer_fe = optim.Adam(model.encoder.parameters(), lr=self.args.lr, weight_decay=self.args.wd) # feature extractor
                optimizer_cls_og = optim.Adam(model.classifier_og.parameters(), lr=self.args.lr, weight_decay=self.args.wd)   
                if self.args.rec:
                    optimizer_ep = optim.Adam(model.decoder.parameters(), lr=self.args.lr , weight_decay=self.args.wd) # edge prediction

                # pretrain encoder & decoder (adopted by GraphSMOTE)
                if self.args.rec:
                    model.train()
                    for epoch in range(self.args.ep_pre):
                        optimizer_fe.zero_grad()
                        optimizer_ep.zero_grad()
                        
                        loss = model(self.features, self.adj, pretrain=True)
                        loss.backward(retain_graph=True)

                        optimizer_fe.step()
                        optimizer_ep.step()

                        if epoch % 100 == 0:
                            print("[Pretrain][Epoch {}] Recon Loss: {}".format(epoch, loss.item()))

                val_f_og = []
                test_results = []

                best_metric = 0.0

                for epoch in trange(self.args.ep):
                    model.train()
                    optimizer_fe.zero_grad()
                    optimizer_cls_og.zero_grad()

                    if self.args.rec:
                        optimizer_ep.zero_grad()
                        loss_nodeclassification, loss_reconstruction = model(self.features, self.adj, labels=self.labels, idx_train=self.idx_train, weight=class_weight, is_og=True)
                        loss = loss_nodeclassification + self.args.rw * loss_reconstruction
                        loss.backward(retain_graph=True)
                        
                        optimizer_fe.step()
                        optimizer_ep.step()
                        optimizer_cls_og.step()

                    else:
                        loss_nodeclassification = model(self.features, labels=self.labels, idx_train=self.idx_train, weight=class_weight, is_og=True)
                        loss = loss_nodeclassification
                        loss.backward(retain_graph=True)
                        
                        optimizer_fe.step()
                        optimizer_cls_og.step()

                    # Evaluation
                    model.eval()
                    embed = model.encoder(self.features)
                    output_original = model.classifier_og(embed)

                    _, macro_F_val, _, bacc_val = utils.performance_measure(output_original[self.idx_val], self.labels[self.idx_val], pre='valid')
                    
                    val_f_og.append(macro_F_val)
                    max_idx = val_f_og.index(max(val_f_og))

                    if best_metric <= macro_F_val:
                        best_metric = macro_F_val
                        best_encoder = deepcopy(model)
                    
                    if (epoch - max_idx > self.args.ep_early) or (epoch+1 == self.args.ep):
                        if epoch - max_idx > self.args.ep_early:
                            print("Early stop")
                        break

                # Save pre-trained encoder
                if self.args.save_encoder:
                    print('Saved pre-trained Encoder')
                    rec_with_ep_pre = 'True_ep_pre_' + str(self.args.ep_pre) + '_rw_' + str(self.args.rw) if self.args.rec else 'False'
                    encoder_info = f'cls_{self.args.cls_og}_cw_{self.args.class_weight}_gamma_{self.args.gamma}_alpha_{self.args.alpha}_lr_{self.args.lr}_dropout_{self.args.dropout}_rec_{rec_with_ep_pre}_seed_{seed}.pkl'
                    if self.args.im_ratio != 1: # manual
                        os.makedirs(f'./pretrained/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}', exist_ok=True)
                        pretrained_encoder = torch.save(best_encoder, f'./pretrained/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}/'+encoder_info)
                    else: # natural
                        os.makedirs(f'./pretrained/natural/{self.args.dataset}', exist_ok=True)
                        pretrained_encoder = torch.save(best_encoder, f'./pretrained/natural/{self.args.dataset}/'+encoder_info)

                model = best_encoder

            # ======================================= Head/Tail Separation & Class Prtotypes ======================================= #
            model.eval()
            embed = model.encoder(self.features)
            prediction = model.classifier_og(embed)
            prediction = torch.softmax(prediction, 1)

            centroids = torch.empty((self.args.nclass, embed.shape[1])).to(embed.device)

            for i, label in enumerate(self.labels.unique()):
                resources = []
                centers = list(map(int, self.idx_train[self.labels[self.idx_train] == label]))
                resources.extend(centers)
                adj_dense = self.adj.to_dense()[centers]
                adj_dense[adj_dense>0] = 1

                similar_matrix = (F.normalize(self.features) @ F.normalize(self.features).T)[centers]
                similar_matrix -= adj_dense

                if self.args.criterion == 'mean':
                    avg_num_candidates = int(sum(self.class_num_mat[:,0]) / len(self.class_num_mat[:,0]))
                elif self.args.criterion == 'median':
                    avg_num_candidates = int(np.median(self.class_num_mat[:,0]))
                elif self.args.criterion == 'max':
                    avg_num_candidates = max(self.class_num_mat[:,0])

                if self.class_num_mat[i,0] < avg_num_candidates:
                    num_candidates_to_fill = avg_num_candidates - self.class_num_mat[i,0]
                    neighbors = np.array(list(set(map(int,self.adj.to_dense()[centers].nonzero()[:,1])) - set(centers)))
                    similar_nodes = np.array(list(set(map(int,similar_matrix.topk(10+1)[1][:,1:].reshape(-1)))))

                    # Candidate Selection
                    candidates_by_neighbors = prediction.cpu()[neighbors, i].sort(descending=True)[1][:num_candidates_to_fill]
                    resource = neighbors[candidates_by_neighbors]
                    if len(candidates_by_neighbors) != 0:
                        resource = [resource] if len(candidates_by_neighbors) == 1 else resource
                        resources.extend(resource)
                    if len(resources) < num_candidates_to_fill:
                            num_candidates_to_fill = num_candidates_to_fill - len(resources)
                            candidates_by_similar_nodes = prediction.cpu()[similar_nodes, i].sort(descending=True)[1][:num_candidates_to_fill]
                            resource = similar_nodes[candidates_by_similar_nodes]
                            if len(candidates_by_similar_nodes) != 0:
                                resource = [resource] if len(candidates_by_similar_nodes) == 1 else resource
                                resources.extend(resource)

                resource = torch.tensor(resources)

                centroids[i, :] = embed[resource].mean(0)
            
            similarity = (F.normalize(embed) @ F.normalize(centroids).t())

            # Top-1 Similarity
            sim_top1_val = torch.argmax(similarity[self.idx_val], 1).long() # top 1 similarity
            sim_top1_test = torch.argmax(similarity[self.idx_test], 1).long() # top 1 similarity

            idx_val_ht_pred = (sim_top1_val >= self.args.sep_point).long()
            idx_test_ht_pred = (sim_top1_test >= self.args.sep_point).long()
            
            idx_class = {}
            for index in [self.idx_val, self.idx_test]:
                idx_class[index] = {}

            idx_class[self.idx_val]['H'] = self.idx_val[(idx_val_ht_pred == 0)].detach().cpu()
            idx_class[self.idx_val]['T'] = self.idx_val[(idx_val_ht_pred == 1)].detach().cpu()

            idx_class[self.idx_test]['H'] = self.idx_test[(idx_test_ht_pred == 0)].detach().cpu()
            idx_class[self.idx_test]['T'] = self.idx_test[(idx_test_ht_pred == 1)].detach().cpu()
            

            # ======================================= Expert Training ======================================= #
            classifier_dict = {}
            for sep in ['HH', 'HT', 'TH', 'TT']:
                idx_train = self.idx_train_set[sep]
                idx_val = self.idx_val_set[sep]
                idx_test = self.idx_test_set[sep]

                best_metric_expert = -1
                max_idx = 0
                val_f_teacher = []
                test_results = []
                
                if sep[1] == 'T':
                    # if degree belongs to tail, finetune head degree classifier
                    classifier = deepcopy(classifier_dict[sep[0] + 'H'])
                else:
                    classifier = model.expert_dict[sep].to(self.args.device)
                optimizer = optim.Adam(classifier.parameters(), lr=self.args.lr_expert, weight_decay=self.args.wd)
                
                for epoch in range(self.args.expert_ep):
                    classifier.train()
                    optimizer.zero_grad()

                    loss = model(self.features, labels=self.labels, idx_train=idx_train, classifier=classifier, sep=sep, is_expert=True)
                    
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Evaluation
                    classifier.eval()
                    output = classifier(embed)

                    acc_val, macro_F_val, gmeans_val, bacc_val = (0,0,0,0) if len(idx_val) ==0 else utils.performance_measure(output[idx_val], self.labels[idx_val], sep_point=self.args.sep_point, sep=sep, pre='valid')
                    
                    val_f_teacher.append(macro_F_val)
                    max_idx = val_f_teacher.index(max(val_f_teacher))

                    if best_metric_expert <= macro_F_val:
                        best_metric_expert = macro_F_val
                        classifier_dict[sep] = deepcopy(classifier) # save best model

                    # Test
                    acc_test, macro_F_test, gmeans_test, bacc_test= (0,0,0,0) if len(idx_test) == 0 else utils.performance_measure(output[idx_test], self.labels[idx_test], sep_point=self.args.sep_point, sep=sep, pre='test')

                    test_results.append([acc_test, macro_F_test, gmeans_test, bacc_test])
                    best_test_result = test_results[max_idx]

                    st = "[seed {}][{}][Expert-{}][Epoch {}]".format(seed, self.args.embedder, sep, epoch)
                    st += "[Val] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}|| ".format(acc_val, macro_F_val, gmeans_val, bacc_val)
                    st += "[Test] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}\n".format(acc_test, macro_F_test, gmeans_test, bacc_test)
                    st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}".format(max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3])
                    
                    if epoch % 100 == 0:
                        print(st)

                    if (epoch - max_idx >= 300) or (epoch + 1 == self.args.ep):
                        if epoch - max_idx >= 300:
                            print('Early Stop!')
                        break
                    

            # ======================================= Student Training ======================================= #
            for sep in ['H', 'T']:
                classifier = model.expert_dict[sep].to(self.args.device)
                optimizer = optim.Adam(classifier.parameters(), lr=self.args.lr_expert, weight_decay=self.args.wd)

                # set idx_train
                idx_train = torch.cat((self.idx_train_set[sep + 'H'], self.idx_train_set[sep + 'T']), 0)
                idx_val = torch.cat((self.idx_val_set[sep + 'H'], self.idx_val_set[sep + 'T']), 0)
                idx_test = torch.cat((self.idx_test_set[sep + 'H'], self.idx_test_set[sep + 'T']), 0)

                best_metric_student = -1
                max_idx = 0
                val_f_student = []
                test_results = []
                
                for epoch in range(self.args.curriculum_ep):
                    classifier.train()
                    optimizer.zero_grad()

                    kd_head, kd_tail, ce_loss = model(self.features, labels=self.labels, idx_train=self.idx_train_set, embed=embed, classifier=classifier, sep=sep, teacher=classifier_dict, is_student=True)
                    alpha = utils.scheduler(epoch, self.args.curriculum_ep)

                    # Head-to-Tail Curriculum Learning
                    loss = ce_loss + (alpha * kd_head + (1-alpha) * kd_tail)

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Evaluation
                    classifier.eval()
                    output = classifier(embed)

                    acc_val, macro_F_val, gmeans_val, bacc_val = (0,0,0,0) if len(idx_val) ==0 else utils.performance_measure(output[idx_val], self.labels[idx_val], sep_point=self.args.sep_point, sep=sep, pre='valid')
                    
                    val_f_student.append(macro_F_val)
                    max_idx = val_f_student.index(max(val_f_student))

                    if best_metric_student <= macro_F_val:
                        best_metric_student = macro_F_val
                        classifier_dict[sep] = deepcopy(classifier) # save best model

                    # Test
                    acc_test, macro_F_test, gmeans_test, bacc_test= (0,0,0,0) if len(idx_test) == 0 else utils.performance_measure(output[idx_test], self.labels[idx_test], sep_point=self.args.sep_point, sep=sep, pre='test')

                    test_results.append([acc_test, macro_F_test, gmeans_test, bacc_test])
                    best_test_result = test_results[max_idx]

                    st = "[seed {}][{}][Student-{}][Epoch {}]".format(seed, self.args.embedder, sep, epoch)
                    st += "[Val] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}|| ".format(acc_val, macro_F_val, gmeans_val, bacc_val)
                    st += "[Test] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}\n".format(acc_test, macro_F_test, gmeans_test, bacc_test)
                    st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}".format(max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3])
                    
                    if epoch % 100 == 0:
                        print(st)

                    if epoch + 1 == self.args.curriculum_ep:
                        break

            # ======================================= Inference Phase =======================================
            final_pred = torch.add(torch.zeros((self.idx_test.shape[0], self.args.nclass)), -999999)
            
            test_to_idx = {}
            for i in range(len(self.idx_test)):
                test = int(self.idx_test[i])
                test_to_idx[test] = i
            
            for sep in ['H', 'T']:
                idx_test = list(map(int,idx_class[self.idx_test][sep]))
                student = classifier_dict[sep]
                student.eval()
                pred = student(embed)

                idx_mapped = list(map(lambda x: test_to_idx[x], idx_test))
                if sep == 'H':
                    final_pred[idx_mapped, 0:self.args.sep_point] = pred[idx_test].cpu()
                elif sep == 'T':
                    final_pred[idx_mapped, self.args.sep_point:self.args.nclass] = pred[idx_test].cpu()

            acc, macro_F, gmeans, bacc = utils.performance_measure(final_pred, self.labels[self.idx_test], pre='test')

            print('=======================================================')
            print('[LTE4G] ACC: {:.1f}, Macro-F1: {:.1f}, GMenas: {:.1f}, bACC: {:.1f}'.format(acc, macro_F, gmeans, bacc))
            print(utils.classification(final_pred, self.labels[self.idx_test].detach().cpu()))
            print(utils.confusion(final_pred, self.labels[self.idx_test].detach().cpu()))
            print(file=text)

            seed_result['acc'].append(float(acc))
            seed_result['macro_F'].append(float(macro_F))
            seed_result['gmeans'].append(float(gmeans))
            seed_result['bacc'].append(float(bacc))
    
        acc = seed_result['acc']
        f1 = seed_result['macro_F']
        gm = seed_result['gmeans']
        bacc = seed_result['bacc']

        print('ACC Macro-F G-Means bACC', file=text)
        print(file=text)
        print('[Averaged result] ACC: {:.1f}+{:.1f}, Macro-F: {:.1f}+{:.1f}, G-Means: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc), np.std(bacc)))
        print('[Averaged result]')
        print('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc), np.std(bacc)), file=text)
        
        logger.info('')
        logger.info(datetime.datetime.now())
        logger.info('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc), np.std(bacc)))
        logger.info(text.name)
        logger.info(f'=================================')

        print(self.args, file=text)
        text.close()
        print(self.args)
         

class modeler(nn.Module):
    def __init__(self, args, adj):
        super(modeler, self).__init__()
        self.args = args
        self.expert_dict = {}

        self.encoder = layers.GNN_Encoder(layer=args.layer, nfeat=args.nfeat, nhid=args.nhid, nhead=args.nhead, dropout=args.dropout, adj=adj)
        if self.args.cls_og == 'GNN': # 'Cora', 'CiteSeer'
            self.classifier_og = layers.GNN_Classifier(layer=args.layer, nhid=args.nhid, nclass=args.nclass, nhead=args.nhead, dropout=args.dropout, adj=adj)
        elif self.args.cls_og == 'MLP': # 'cora_full'
            self.classifier_og = layers.MLP(nhid=args.nhid, nclass=args.nclass)
        
        for sep in ['HH', 'H', 'TH', 'T']:
            num_class = args.sep_point if sep[0] == 'H' else args.nclass - args.sep_point
            self.expert_dict[sep] = layers.GNN_Classifier(layer=args.layer, nhid=args.nhid, nclass=num_class, nhead=args.nhead, dropout=args.dropout, adj=adj)
        
        if self.args.rec:
            self.decoder = layers.Decoder(nhid=args.nhid, dropout=args.dropout)

    def forward(self, features, adj=None, labels=None, idx_train=None, classifier=None, embed=None, sep=None, teacher=None, pretrain=False, weight=None, is_og=False, is_expert=False, is_student=False):
        if embed == None:
            embed = self.encoder(features)

        if pretrain:
            generated_G = self.decoder(embed)
            loss_reconstruction = utils.adj_mse_loss(generated_G, adj.detach().to_dense())
            return loss_reconstruction
            
        if is_og:
            output = self.classifier_og(embed)
            if self.args.class_weight:
                ce_loss = -F.cross_entropy(output[idx_train], labels[idx_train], weight=weight)
                pt = torch.exp(-F.cross_entropy(output[idx_train], labels[idx_train]))
                loss_nodeclassfication = -((1 - pt) ** self.args.gamma) * ce_loss
            else:
                ce_loss = -F.cross_entropy(output[idx_train], labels[idx_train])
                pt = torch.exp(-F.cross_entropy(output[idx_train], labels[idx_train]))
                loss_nodeclassfication = -((1 - pt) ** self.args.gamma) * self.args.alpha * ce_loss
            
            if self.args.rec:
                generated_G = self.decoder(embed)
                loss_reconstruction = utils.adj_mse_loss(generated_G, adj.detach().to_dense())
                return loss_nodeclassfication, loss_reconstruction
            else:
                return loss_nodeclassfication
        
        if is_expert:
            pred = classifier(embed)

            if sep in ['T', 'TH', 'TT']:
                labels = labels - self.args.sep_point

            loss_nodeclassfication = F.cross_entropy(pred[idx_train], labels[idx_train])

            return loss_nodeclassfication

        if is_student:
            # teacher
            teacher_head_degree = teacher[sep+'H']
            teacher_tail_degree = teacher[sep+'T']
            idx_train_head_degree = idx_train[sep+'H']
            idx_train_tail_degree = idx_train[sep+'T']
            idx_train_all = torch.cat((idx_train_head_degree, idx_train_tail_degree), 0)

            teacher_head_degree.eval()
            teacher_tail_degree.eval()
            
            out_head_teacher = teacher_head_degree(embed)[idx_train_head_degree]
            out_tail_teacher = teacher_tail_degree(embed)[idx_train_tail_degree]
                
            # student
            out_head_student = classifier(embed)[idx_train_head_degree]
            out_tail_student = classifier(embed)[idx_train_tail_degree]

            kd_head = F.kl_div(F.log_softmax(out_head_student / self.args.T, dim=1), F.softmax(out_head_teacher / self.args.T, dim=1), reduction='mean') * self.args.T * self.args.T
            kd_tail = F.kl_div(F.log_softmax(out_tail_student / self.args.T, dim=1), F.softmax(out_tail_teacher / self.args.T, dim=1), reduction='mean') * self.args.T * self.args.T
            
            if sep in ['T', 'TH', 'TT']:
                labels = labels - self.args.sep_point

            ce_loss = F.cross_entropy(classifier(embed)[idx_train_all], labels[idx_train_all])

            return kd_head, kd_tail, ce_loss