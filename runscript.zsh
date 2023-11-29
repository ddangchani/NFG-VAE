# Script to run experiments
git pull origin run

# python train.py --dependence_type=1 --node_size=100 --seed=42 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=50
# python train.py --dependence_type=1 --node_size=100 --seed=42 --flow_type='DAGGNN' --dependence_prop=0.5 --k_max_iter=50
python train.py --dependence_type=1 --node_size=100 --seed=42 --flow_type='HF' --dependence_prop=0.5 --k_max_iter=50

git pull origin run
git add .
git commit -m "dependence_type=1, node_size=20, seed=1 to 5, dependence_prop=0.5, flow_type=DAGGNN,IAF"
git push origin run