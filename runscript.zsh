# Script to run experiments
git pull origin run

python train.py --dependence_type=1 --node_size=100 --seed=42 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=50 --lagrange=0 --tau_A=0.1
python train.py --dependence_type=1 --node_size=100 --seed=42 --flow_type='DAGGNN' --dependence_prop=0.5 --k_max_iter=50 --lagrange=1 --tau_A=0.1

git pull origin run
git add .
git commit -m "dependece_type=1, node_size=100, seed=42, flow_type=IAF,DAGGNN, dependence_prop=0.5, k_max_iter=50, lagrange=0, tau_A=0.1"
git push origin run