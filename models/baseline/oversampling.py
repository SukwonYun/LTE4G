from embedder import embedder
import torch.nn as nn
import layers
import torch.optim as optim
import utils
import torch.nn.functional as F
import torch
import numpy as np
from copy import deepcopy
import os

class oversampling():
    def __init__(self, args):
        self.args = args

    def training(self):
        self.args.embedder = f'({self.args.layer.upper()})' + self.args.embedder + f'_cls_{self.args.cls_og}' + f'_im_class_num_{self.args.im_class_num}' 
        if self.args.im_ratio == 1: # natural
            os.makedirs(f'./results/baseline/natural/{self.args.dataset}', exist_ok=True)
            text = open(f'./results/baseline/natural/{self.args.dataset}/{self.args.embedder}.txt', 'w')
        else: # manual
            os.makedirs(f'./results/baseline/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}', exist_ok=True)
            text = open(f'./results/baseline/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}/{self.args.embedder}.txt', 'w')

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
            self.adj, self.features, self.labels, self.idx_train = \
            src_upsample(self.adj, self.features, self.labels, self.idx_train, portion=self.args.up_scale, im_class_num=self.im_class_num)

            model = modeler(self.args, self.adj).to(self.args.device)
            optimizer_fe = optim.Adam(model.encoder.parameters(), lr=self.args.lr, weight_decay=self.args.wd)  # feature extractor
            optimizer_cls = optim.Adam(model.classifier.parameters(), lr=self.args.lr, weight_decay=self.args.wd)  # node classifier

            # Main training
            val_f = []
            test_results = []

            best_metric = 0

            for epoch in range(self.args.ep):
                model.train()
                optimizer_fe.zero_grad()
                optimizer_cls.zero_grad()

                loss = model(self.features, self.labels, self.idx_train)

                loss.backward()

                optimizer_fe.step()
                optimizer_cls.step()

                # Evaluation
                model.eval()
                embed = model.encoder(self.features)
                output = model.classifier(embed)

                acc_val, macro_F_val, gmeans_val, bacc_val = utils.performance_measure(output[self.idx_val], self.labels[self.idx_val], pre='valid')

                val_f.append(macro_F_val)
                max_idx = val_f.index(max(val_f))

                if best_metric <= macro_F_val:
                    best_metric = macro_F_val
                    best_model = deepcopy(model)

                # Test
                acc_test, macro_F_test, gmeans_test, bacc_test= utils.performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')

                test_results.append([acc_test, macro_F_test, gmeans_test, bacc_test])
                best_test_result = test_results[max_idx]

                st = "[seed {}][{}][Epoch {}]".format(seed, self.args.embedder, epoch)
                st += "[Val] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}|| ".format(acc_val, macro_F_val, gmeans_val, bacc_val)
                st += "[Test] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}\n".format(acc_test, macro_F_test, gmeans_test, bacc_test)
                st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}".format(max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3])
                    
                if epoch % 100 == 0:
                    print(st)

                if (epoch - max_idx > self.args.ep_early) or (epoch+1 == self.args.ep):
                    if epoch - max_idx > self.args.ep_early:
                        print("Early stop")
                    embed = best_model.encoder(self.features)
                    output = best_model.classifier(embed)
                    best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3] = utils.performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')
                    print("[Best Test Result] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}".format(best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3]),file=text)
                    print(utils.classification(output[self.idx_test], self.labels[self.idx_test].detach().cpu()), file=text)
                    print(utils.confusion(output[self.idx_test], self.labels[self.idx_test].detach().cpu()), file=text)
                    print(file=text)
                    break

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['macro_F'].append(float(best_test_result[1]))
            seed_result['gmeans'].append(float(best_test_result[2]))
            seed_result['bacc'].append(float(best_test_result[3]))

        acc = seed_result['acc']
        f1 = seed_result['macro_F']
        gm = seed_result['gmeans']
        bacc = seed_result['bacc']

        print('[Averaged result] ACC: {:.1f}+{:.1f}, Macro-F: {:.1f}+{:.1f}, G-Means: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc), np.std(bacc)))
        print(file=text)
        print('ACC Macro-F G-Means bACC', file=text)
        print('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc), np.std(bacc)), file=text)
        print(file=text)
        print(self.args, file=text)
        print(self.args)
        text.close()

class modeler(nn.Module):
    def __init__(self, args, adj):
        super(modeler, self).__init__()
        self.args = args

        self.encoder = layers.GNN_Encoder(layer=args.layer, nfeat=args.nfeat, nhid=args.nhid, nhead=args.nhead, dropout=args.dropout, adj=adj)
        if args.cls_og == 'GNN':
            self.classifier = layers.GNN_Classifier(layer=args.layer, nhid=args.nhid, nclass=args.nclass, nhead=args.nhead, dropout=args.dropout, adj=adj)
        elif args.cls_og == 'MLP':
            self.classifier = layers.MLP(nhid=args.nhid, nclass=args.nclass)

    def forward(self, features, labels, idx_train, pretrain=False):
        embed = self.encoder(features)
        output = self.classifier(embed)

        loss_nodeclassification = F.cross_entropy(output[idx_train], labels[idx_train])

        return loss_nodeclassification

def src_upsample(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0: #refers to even distribution
            avg_number = int(idx_train.shape[0] / (c_largest + 1))
            c_portion = int(avg_number/new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
            
            num = int(new_chosen.shape[0]*portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train