# Script to run experiments
git pull origin train_modification

python train.py --dependence_type=1 --node_size=50 --seed=21 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=50 --seed=31 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=50 --seed=41 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'
python train.py --dependence_type=1 --node_size=50 --seed=51 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='DAGGNN'

git pull origin train_modification
git add .
git commit -m "optimization test run DAGGNN"
git push origin train_modification