# Script to run experiments
git pull origin train_modification

python train.py --dependence_type=0 --node_size=30 --seed=2 --flow_type='ccIAF' --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=2 --flow_type='DAGGNN' --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --graph_dist='laplace'


git pull origin train_modification
git add .
git commit -m "dependece_type=0, node_size=30, seed=2, flow_type=IAF, k_max_iter=50, lagrange=1"
git push origin train_modification