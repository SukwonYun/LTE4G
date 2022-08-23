# LTE4G: Long-Tail Experts for Graph Neural Networks

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://www.cikm2022.org/" alt="Conference">
        <img src="https://img.shields.io/badge/CIKM'22-brightgreen" /></a>
</p>

The official source code for **[LTE4G: Long-Tail Experts for Graph Neural Networks](https://arxiv.org/abs/2208.10205)** paper, accepted at CIKM 2022.


### Overview
Existing Graph Neural Networks (GNNs) usually assume a balanced situation where both the class distribution and the node degree
distribution are balanced. However, in real-world situations, we often encounter cases where a few classes (i.e., head class) dominate other classes (i.e., tail class) as well as in the node degree perspective, and thus naively applying existing GNNs eventually fall short of generalizing to the tail cases. Although recent studies proposed methods to handle long-tail situations on graphs, they only focus on either the class long-tailedness or the degree long-tailedness. In this paper, we propose a novel framework for training GNNs, called Long-Tail Experts for Graphs (**LTE4G**), which jointly considers the class long-tailedness, and the degree long-tailedness for node classification. The core idea is to assign an expert GNN model to each subset of nodes that are split in a balanced manner considering both the class and degree long-tailedness. After having trained an expert for each balanced subset, we adopt knowledge distillation to obtain two class-wise students, i.e., Head class student and Tail class student, each of which is responsible for classifying nodes in the head classes and tail classes, respectively. We demonstrate that **LTE4G** outperforms a wide range of state-of-the-art methods in node classification evaluated on both manual and natural imbalanced graphs. 

<img src="https://user-images.githubusercontent.com/68312164/185851453-53771970-6f06-4b64-a7e3-a70766f8c41a.png">

### Requirements
- python version: 3.7.11
- numpy version: 1.19.2
- pytorch version: 1.8.0
- torch-geometric version: 2.0.1

To set environment that perfectly matches the paper, please run: 
```
conda env create -f environment.yml
conda activate py37
```

### Hyperparameters
Following Options can be passed to `main.py`

`--dataset:` Name of the dataset. Cora, CiteSeer, and cora_full has been used in the paper.  
usage example :`--dataset Cora`

`--im_class_num:`
Number of minority(i.e., tail) classes.  
usage example :`--im_class_num 3`

`--im_ratio:`
Ratio between the number of samples in the Tail class and the Head class (i.e., Tail/Head) 1 for natural, [0.2, 0.1, 0.05] for manual, 0.01 for LT setting  
usage example :`--im_ratio 0.1`

`--sep_class:`
Separation of Head and Tail class. (Top $p$ for Head and $1-p$ for Tail)  
usage example :`--sep_class pareto_73`

`--sep_degree:`
Separation of Head and Tail degree. (Head: above 6, Tail: below 5.)  
usage example :`--sep_degree 5`

`--layer:`
GNN layer, e.g., GCN and GAT  
usage example :`--layer gcn`

`--rec:`
Whether to include edge reconstruction process suggested in GraphSMOTE.  
usage example :`--rec True`

`--gamma:`
Focusing parameter on focal loss.  
usage example :`--gamma 1`

`--alpha:`
Weight factor on focal loss.  
usage example :`--alpha 0.9`

`--class_weight:`
Wheter to apply inverse cardinality as a weight factor on focal loss.  
usage example :`--class_weight False`


### How to Run

You can run the model with following options
- To reproduce Table 3 (Manual Setting) in paper
```
sh ./sh/best_cora_3_0.1.sh
```

- or you can run the file with above mentioned hyperparameters
```
python main.py --gpu 0 --dataset Cora --im_class_num 3 --im_ratio 0.1 --layer gat --rec True --gamma 1 --alpha 0.9 --class_weight False
```

- Otherwise, to run baselines in paper, add `--baseline`
```
python main.py --baseline --embedder origin
```