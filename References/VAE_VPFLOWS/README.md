# VAE with Volume-Preserving Flows
This is a PyTorch implementation of two volume-preserving flows as described in the following papers:
* Tomczak, J. M., & Welling, M., Improving Variational Auto-Encoders using Householder Flow, [arXiv preprint](https://arxiv.org/abs/1611.09630), 2016
* Tomczak, J. M., & Welling, M., Improving Variational Auto-Encoders using convex combination linear Inverse Autoregressive Flow, [arXiv preprint](https://arxiv.org/abs/1706.02326), 2017

## Data
The experiments can be run on four datasets:
* static MNIST: links to the datasets can found at [link](https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST);
* binary MNIST: the dataset is loaded from [Keras](https://keras.io/);
* OMNIGLOT: the dataset could be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset could be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).

## Run the experiment
1. Set-up your experiment in `experiment.py`.
2. Run experiment:
```bash
python experiment.py
```
## Models
You can run a vanilla VAE, a VAE with the Householder Flow (HF) or the convex combination linear Inverse Autoregressive Flow (ccLinIAF) by setting `model_name` argument to either `vae`, `vae_HF` or `vae_ccLinIAF`, respectively. Setting `number_combination` for `vae_ccLinIAF` to 1 results in `vae_linIAF`.

## Citation

Please cite our paper if you use this code in your research:

```
@article{TW:2017,
  title={{Improving Variational Auto-Encoders using convex combination linear Inverse Autoregressive Flow}},
  author={Tomczak, Jakub M and Welling, Max},
  journal={arXiv},
  year={2017}
}
```

## Acknowledgments
The research conducted by Jakub M. Tomczak was funded by the European Commission within the Marie Skłodowska-Curie Individual Fellowship (Grant No. 702666, ”Deep learning and Bayesian inference for medical imaging”).
