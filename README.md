# Learning semi-Markovian DAGs with flow-based VAE

- Project proposed for M1399_000400-2023fall(Deep Learning) at Seoul National University

## Overview

- Learning Non-Gaussian linear SEMs with independent noise using IAF based VAE
- Learning linear SEMs with dependent noise (i.e. semi-Markovian graph) using IAF based VAE


## Environment

- Python 3.8
- Pytorch 2.1.0
- networkx 2.8.7

## Usage

To train the model, run the following command

- Semi-Markovian DAGs
```bash
python train.py --dependence_type=1 --dependence_prop=0.3 --node_size=20 --seed=123 --flow_type='IAF'
```

- Non-Gaussian DAGs
```bash
python train.py --dependence_type=0 --graph_dist='laplace' --node_size=20 --seed=123 --flow_type='IAF'
```
> Possible noise_dist : 'normal', 'uniform', 'exponential', 'laplace', 'gumbel'


## Code Sources
Code benefit from the following works
- [DAG NOTEARS](https://github.com/xunzheng/notears)
- [DAG-GNN](https://github.com/fishmoon1234/DAG-GNN)
- [Flow-VAE](https://github.com/fmu2/flow-VAE)
- [IAF-VAE](https://github.com/pclucas14/iaf-vae)
- [VAE-VPFLOWS](https://github.com/jmtomczak/vae_vpflows)
