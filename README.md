# NFG-VAE : Normalizing Flow based Graph Learning with Variational Autoencoder

- Project proposed for M1399_000400-2023fall(Deep Learning) at Seoul National University

## Topic

- Learning Non-Gaussian DAGs with Variational Autoencoder
- Learning Linear SEM with dependent noise using Normalizing Flow


## Environment

- Python 3.8
- Pytorch 2.1.0

## Structure

- `References/` : Codes used for this project
- `Tex/` : Latex files for report
- `model.py` : VAE, NF code
- `train.py` : main code for run model training
- `data.py` : code for generating data
- `utils.py` : code for evaluation(metric)

## Code Sources

- Code benefit from the following works
  - [DAG NOTEARS](https://github.com/xunzheng/notears)
  - [DAG-GNN](https://github.com/fishmoon1234/DAG-GNN)
  - [Flow-VAE](https://github.com/fmu2/flow-VAE)
  - [IAF-VAE](https://github.com/pclucas14/iaf-vae)
