# Script to run experiments
git pull origin train_modification

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.016 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

python train.py --dependence_type=1 --node_size=20 --seed=11 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=21 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=31 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=41 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=20 --seed=51 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

git pull origin train_modification
git add .
git commit -m "run DAGGNN vs IAF 20 nodes"
git push origin train_modification