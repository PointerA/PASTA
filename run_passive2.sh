#!/bin/bash

conda_env="iwqos"

source activate $conda_env

cd ~/iwqos/



# nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 3 >> "./log/cifar10/fedsgd3_1.log" 2>&1 &
# sleep 2
# nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 4 >> "./log/cifar10/fedsgd4_1.log" 2>&1 &
# sleep 2

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 3 >> "./log/cifar10/fedsgd3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 4 >> "./log/cifar10/fedsgd4_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 3 >> "./log/cifar10/fedsgd3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 4 >> "./log/cifar10/fedsgd4_1.log" 2>&1 &
sleep 2





sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 3 >> "./log/cifar10/our3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 4 >> "./log/cifar10/our4_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 3 >> "./log/cifar10/our3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 4 >> "./log/cifar10/our4_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 3 >> "./log/cifar10/our3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 4 >> "./log/cifar10/our4_1.log" 2>&1 &
sleep 2





sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FedBCD3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FedBCD4_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FedBCD3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FedBCD4_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FedBCD3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FedBCD4_1.log" 2>&1 &
sleep 2



sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FlexVFL3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FlexVFL4_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FlexVFL3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FlexVFL4_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FlexVFL3_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FlexVFL4_1.log" 2>&1 &
sleep 2