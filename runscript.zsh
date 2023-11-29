# Script to run experiments
git pull origin train_modification

python train.py --dependence_type=1 --node_size=30 --seed=5 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=50 --lagrange=1


git pull origin train_modification
git add .
git commit -m "dependece_type=1, node_size=30, seed=3to5, flow_type=IAF,DAGGNN, dependence_prop=0.5, k_max_iter=50, lagrange=0, tau_A=0.1"
git push origin train_modification