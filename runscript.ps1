# Script to run experiments
python train.py --dependence_type=1 --node_size=50 --seed=3 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=100
python train.py --dependence_type=1 --node_size=50 --seed=4 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=100
python train.py --dependence_type=1 --node_size=100 --seed=5 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=100
python train.py --dependence_type=1 --node_size=100 --seed=6 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=100
python train.py --dependence_type=1 --node_size=100 --seed=7 --flow_type='IAF' --dependence_prop=0.5 --k_max_iter=100


git pull origin run
git add .
git commit -m "dependence_type=1, node_size=50, seed=1 to 5, dependence_prop=0.5, flow_type=IAF"
git push origin run