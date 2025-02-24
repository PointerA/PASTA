#!/bin/bash

conda_env="iwqos"

source activate $conda_env

cd ~/iwqos/
# 


# nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 5 >> "./log/cifar10/fedsgd5_1.log" 2>&1 &
# sleep 2
# nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 6 >> "./log/cifar10/fedsgd6_1.log" 2>&1 &
# sleep 2

# sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 5 >> "./log/cifar10/fedsgd5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 6 >> "./log/cifar10/fedsgd6_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 5 >> "./log/cifar10/fedsgd5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 6 >> "./log/cifar10/fedsgd6_1.log" 2>&1 &
sleep 2





sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 5 >> "./log/cifar10/our5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 6 >> "./log/cifar10/our6_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 5 >> "./log/cifar10/our5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 6 >> "./log/cifar10/our6_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 5 >> "./log/cifar10/our5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 6 >> "./log/cifar10/our6_1.log" 2>&1 &
sleep 2






sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FedBCD5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FedBCD6_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FedBCD5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FedBCD6_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FedBCD5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FedBCD6_1.log" 2>&1 &
sleep 2





sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FlexVFL5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FlexVFL6_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FlexVFL5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FlexVFL6_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FlexVFL5_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FlexVFL6_1.log" 2>&1 &
sleep 2