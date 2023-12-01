# Script to run experiments
git pull origin train_modification

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.004 --loss_prevent=0 --logits=0

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.1 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.3 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

python train.py --dependence_type=1 --node_size=10 --seed=11 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=21 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=31 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=41 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=10 --seed=51 --dependence_prop=0.8 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

git pull origin train_modification
git add .
git commit -m "run DAGGNN vs IAF 10 nodes"
git push origin train_modification