#!/bin/bash

conda_env="iwqos"



source activate $conda_env
cd ~/iwqos/

# 
# nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 7 >> "./log/cifar10/fedsgd7_1.log" 2>&1 &
# sleep 12000
# 
# nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 7 >> "./log/cifar10/fedsgd7_1.log" 2>&1 &

# sleep 12000
# 
# nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 7 >> "./log/cifar10/our7_1.log" 2>&1 &
# sleep 12000
# 
# nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 7 >> "./log/cifar10/our7_1.log" 2>&1 &
# sleep 12000
# 
# nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 7 >> "./log/cifar10/our7_1.log" 2>&1 &


# sleep 7000
# 
# nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FedBCD7_1.log" 2>&1 &
# sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FedBCD7_1.log" 2>&1 &
sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FedBCD7_1.log" 2>&1 &

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FlexVFL7_1.log" 2>&1 &
sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FlexVFL7_1.log" 2>&1 &
sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FlexVFL7_1.log" 2>&1 &