import argparse

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true', default=False, help="Run LTE4G or baseline")
    parser.add_argument('--gpu', type=int, default=0, help="Choose GPU number")
    parser.add_argument('--dataset', type=str, default='cora_full', choices=['Cora', 'CiteSeer', 'cora_full'])
    parser.add_argument('--im_class_num', type=int, default=3, help="Number of tail classes")
    parser.add_argument('--im_ratio', type=float, default=1, help="1 for natural, [0.2, 0.1, 0.05] for manual, 0.01 for LT setting")
    parser.add_argument('--layer', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rw', type=float, default=0.000001, help="Balances edge loss within node classification loss")
    parser.add_argument('--ep_pre', type=int, default=50, help="Number of epochs to pretrain.")
    parser.add_argument('--ep', type=int, default=10000, help="Number of epochs to train.")
    parser.add_argument('--ep_early', type=int, default=1000, help="Early stop criterion.")
    parser.add_argument('--add_sl', type=str2bool, default=True, help="Whether to include self-loop")
    parser.add_argument('--adj_norm_1', action='store_true', default=True, help="D^(-1)A")
    parser.add_argument('--adj_norm_2', action='store_true', default=False, help="D^(-1/2)AD^(-1/2)")
    parser.add_argument('--nhid', type=int, default=64, help="Number of hidden dimensions")
    parser.add_argument('--nhead', type=int, default=1, help="Number of multi-heads")
    parser.add_argument('--wd', type=float, default=5e-4, help="Controls weight decay")
    parser.add_argument('--num_seed', type=int, default=3, help="Number of total seeds") 
    parser.add_argument('--is_normalize', action='store_true', default=False, help="Normalize features")
    parser.add_argument('--cls_og', type=str, default='GNN', choices=['GNN', 'MLP'], help="Wheter to user (GNN+MLP) or (MLP) as a classifier")

    if not parser.parse_known_args()[0].baseline:
        parser.add_argument('--embedder', nargs='?', default='lte4g')
        parser.add_argument('--rec', type=str2bool, default=False, help="Whether to include edge reconstruction") # key hyperparameter
        parser.add_argument('--lr_expert', type=float, default=0.01, help="Learning Rate for Expert.") 
        parser.add_argument('--criterion', type=str, default='mean', choices=['max', 'mean', 'median'], help="criterion for number of samples for each class's prototype")
        parser.add_argument('--sep_class', type=str, default='pareto_73', help="top (p) for Head and (1-p) for Tail.") # key hyperparameter
        parser.add_argument('--sep_degree', type=int, default=5, help="Criteria of Head / Tail Degree separation")
        parser.add_argument('--class_weight', type=str2bool, default=True, help="Wheter to apply inverse cardinality as a class weight.") # key hyperparameter
        parser.add_argument('--gamma', type=float, default=1, help="Focusing parameter `\gamma >= 0") # key hyperparameter
        parser.add_argument('--alpha', type=float, default=0.6, help="Weighting factor `\alpha \in [0, 1]") # key hyperparameter
        parser.add_argument('--T', type=float, default=1, help="Temperature of Knowledge Distillation loss.")
        parser.add_argument('--expert_ep', type=int, default=1000, help="Number of epochs while obtaining HH/HT/TH/TT experts.")
        parser.add_argument('--curriculum_ep', type=int, default=500, help='Number of epochs while obtaining Head/Tail students.')
        parser.add_argument('--pretrained_encoder', type=str2bool, default=False, help="Whether to use pretrained encoder or not")
        parser.add_argument('--save_encoder', type=str2bool, default=False, help="Whether to save pretrained encoder or not")

    elif parser.parse_known_args()[0].baseline:
        parser.add_argument('--embedder', nargs='?', default='origin', choices=['origin', 'reweight', 'oversampling', 'smote', 'embed_smote', 'graphsmote_T', 'graphsmote_O'])
        parser.add_argument('--up_scale', type=float, default=1, help="Scale of Oversampling")

    return parser