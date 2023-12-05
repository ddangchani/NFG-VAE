# Script to run experiments
git pull origin train_modification

python train.py --dependence_type=0 --node_size=20 --seed=101 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=201 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=301 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=401 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=501 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0 --graph_dist='laplace'

python train.py --dependence_type=0 --node_size=20 --seed=101 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=201 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=301 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=401 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=20 --seed=501 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'

git pull origin train_modification
git add .
git commit -m "run IAF vs DAGGNN 20 nodes independent laplace"
git push origin train_modification

python train.py --dependence_type=0 --node_size=30 --seed=101 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.036 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=201 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.036 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=301 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.036 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=401 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.036 --loss_prevent=0 --logits=0 --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=501 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.036 --loss_prevent=0 --logits=0 --graph_dist='laplace'

python train.py --dependence_type=0 --node_size=30 --seed=101 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=201 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=301 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=401 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'
python train.py --dependence_type=0 --node_size=30 --seed=501 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN' --graph_dist='laplace'


git pull origin train_modification
git add .
git commit -m "run IAF vs DAGGNN 30 nodes independent laplace"
git push origin train_modification