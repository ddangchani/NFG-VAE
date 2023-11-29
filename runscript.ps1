# Script to run experiments
python train.py --dependence_type=1 --node_size=10 --seed=3 --flow_type='IAF' --dependence_prop=0.5
python train.py --dependence_type=1 --node_size=10 --seed=4 --flow_type='IAF' --dependence_prop=0.5
python train.py --dependence_type=1 --node_size=10 --seed=5 --flow_type='IAF' --dependence_prop=0.5
python train.py --dependence_type=1 --node_size=10 --seed=6 --flow_type='IAF' --dependence_prop=0.5


git pull origin run
git add .
git commit -m "dependence_type=1, node_size=10, seed=1 to 5, dependence_prop=0.5, flow_type=DAGGNN,IAF"
git push origin run