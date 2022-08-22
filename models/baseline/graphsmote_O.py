from embedder import embedder
import torch.nn as nn
import layers
import torch.optim as optim
import utils
import torch.nn.functional as F
import torch
import numpy as np
from copy import deepcopy
import data_load
import os
from scipy.spatial.distance import pdist,squareform
import random

class graphsmote_O():
    def __init__(self, args):
        self.args = args

    def training(self):
        self.args.embedder = f'({self.args.layer.upper()})' + self.args.embedder + '_im_class_num_' + str(self.args.im_class_num) 
        if self.args.ep_pre != 0:
            self.args.embedder += f'_ep_pre_{self.args.ep_pre}'

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
            self = embedder(self.args)
            print('seed:', seed, file=text)

            model = modeler(self.args, self.adj).to(self.args.device)
            optimizer_fe = optim.Adam(model.encoder.parameters(), lr=self.args.lr, weight_decay=self.args.wd)  # feature extractor
            optimizer_ep = optim.Adam(model.decoder.parameters(), lr=self.args.lr, weight_decay=self.args.wd)  # edge predictor
            optimizer_cls = optim.Adam(model.classifier.parameters(), lr=self.args.lr, weight_decay=self.args.wd)  # node classifier

            
            # pretrain
            pretrain_losses = []
            for epoch in range(self.args.ep_pre):
                model.train()
                optimizer_fe.zero_grad()
                optimizer_ep.zero_grad()
                optimizer_cls.zero_grad()

                loss = model(self.features, self.adj, self.labels, self.idx_train, pretrain=True)
                loss.backward()

                optimizer_fe.step()
                optimizer_ep.step()

                if epoch % 100 == 0:
                    print("[Pretrain][Epoch {}] Recon Loss: {}".format(epoch, loss.item()))

                pretrain_losses.append(loss.item())
                min_idx = pretrain_losses.index(min(pretrain_losses))
                if epoch - min_idx > 500:
                    print("Pretrain converged")
                    break
                
            # Main training
            val_f = []
            test_results = []

            best_metric = 0

            for epoch in range(self.args.ep):
                model.train()
                optimizer_fe.zero_grad()
                optimizer_cls.zero_grad()
                optimizer_ep.zero_grad()

                loss_reconstruction, loss_nodeclassification = model(self.features, self.adj, self.labels, self.idx_train)

                loss = loss_nodeclassification + self.args.rw * loss_reconstruction
                loss.backward()

                optimizer_fe.step()
                optimizer_cls.step()
                optimizer_ep.step()

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
        self.classifier = layers.GNN_Classifier(layer=args.layer, nhid=args.nhid, nclass=args.nclass, nhead=args.nhead, dropout=args.dropout, adj=adj)

        self.decoder = layers.Decoder(nhid=args.nhid, dropout=args.dropout)

    def forward(self, features, adj, labels, idx_train, pretrain=False):
        embed = self.encoder(features)

        ori_num = labels.shape[0]
        embed, labels_new, idx_train_new, adj_up = recon_upsample(embed, labels, idx_train, adj=adj.detach().to_dense(), portion=self.args.up_scale, im_class_num=self.args.im_class_num)
        generated_G = self.decoder(embed)  # generate edges

        loss_reconstruction = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())  # Equation 6

        if pretrain:
            return loss_reconstruction

        adj_new = generated_G ##
        adj_new = torch.mul(adj_up, adj_new)  ###

        adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()
        adj_new = adj_new.detach() ##

        # adj_new[adj_new != 0] = 1
        if self.args.adj_norm_1:
            adj_new = utils.normalize_adj(adj_new.to_sparse())

        elif self.args.adj_norm_2:
            adj_new = utils.normalize_sym(adj_new.to_sparse())

        output = self.classifier(embed, adj_new)
        loss_nodeclassification = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])  # Equation 11

        return loss_reconstruction, loss_nodeclassification

def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            avg_number = int(idx_train.shape[0] / (c_largest + 1))
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1) # Equation 3
            
            interp_place = random.random()
            new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place # Equation 4


            new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(c_largest-i)
            idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed,new_embed), 0)
            labels = torch.cat((labels,new_labels), 0)
            idx_train = torch.cat((idx_train,idx_train_append), 0)

            ## The generated edges are only from those that were originally edges of the sampled nodes and the nearest neighbor of the sampled nodes
            if adj is not None:
                adj[adj != 0] = 1
                if adj_new is None:
                    # adj_new = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0)) #?
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[chosen[idx_neighbor], :], min=0.0, max=1.0))  # ?
                else:
                    # temp = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[chosen[idx_neighbor], :], min=0.0, max=1.0))  # ?
                    adj_new = torch.cat((adj_new, temp), 0)
            ##

    # return embed, labels, idx_train

    ##
    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train
    ##