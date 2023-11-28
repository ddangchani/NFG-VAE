# Script to run experiments
git pull origin run

python train.py --dependence_type=1 --node_size=10 --seed=2 --flow_type='IAF' --dependence_prop=0.3
python train.py --dependence_type=1 --node_size=10 --seed=3 --flow_type='IAF' --dependence_prop=0.3
python train.py --dependence_type=1 --node_size=10 --seed=4 --flow_type='IAF' --dependence_prop=0.3
python train.py --dependence_type=1 --node_size=10 --seed=5 --flow_type='IAF' --dependence_prop=0.3

git add .
git commit -m "dependence_type=1, node_size=10, seed=1 to 5, dependence_prop=0.3, flow_type=DAGGNN,IAF"
git push origin run