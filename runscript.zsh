# Script to run experiments
git pull origin train_modification

python train.py --dependence_type=1 --node_size=30 --seed=4 --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --flow_type='IAF' --tau_A=0.1 --loss_prevent=0 --logits=0

git pull origin train_modification
git add .
git commit -m "optimization test run"
git push origin train_modification