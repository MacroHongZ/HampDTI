# HampDTI
A heterogeneous graph automatic meta-path learning method for drug-target interaction prediction

# Requirements
- Pytorch 1.7.0
- Numpy 1.19.2
- Rdkit
- pyG

# Quick start

To reproduce our results:  please run `main.py`. Options are:

`--num_channels: the number of channels, default: 4`

`--num_layers: the maximum length of the learned meta-paths, default: 3`

`--alpha: hyperparameter in loss function, default: 0.4`

`-t: select experimental data, default: 'o'`

