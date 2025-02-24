#!/bin/bash

conda_env="iwqos"

source activate $conda_env

cd ~/iwqos/


nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 0 >> "./log/cifar10/fedsgd0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 1 >> "./log/cifar10/fedsgd1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 2 >> "./log/cifar10/fedsgd2_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 0 >> "./log/cifar10/fedsgd0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 1 >> "./log/cifar10/fedsgd1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 2 >> "./log/cifar10/fedsgd2_1.log" 2>&1 &
sleep 2


#our

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 0 >> "./log/cifar10/our0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 1 >> "./log/cifar10/our1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 2 >> "./log/cifar10/our2_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 0 >> "./log/cifar10/our0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 1 >> "./log/cifar10/our1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 2 >> "./log/cifar10/our2_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 0 >> "./log/cifar10/our0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 1 >> "./log/cifar10/our1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 2 >> "./log/cifar10/our2_1.log" 2>&1 &
sleep 2

#FedBCD

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FedBCD0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FedBCD1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FedBCD2_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FedBCD0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FedBCD1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FedBCD2_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FedBCD0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FedBCD1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FedBCD2_1.log" 2>&1 &
sleep 2

#FlexVFL

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FlexVFL0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FlexVFL1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FlexVFL2_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FlexVFL0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FlexVFL1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 98 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FlexVFL2_1.log" 2>&1 &
sleep 2

sleep 12000

nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FlexVFL0_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FlexVFL1_1.log" 2>&1 &
sleep 2
nohup python -u main_pipeline_asyn.py --seed 99 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FlexVFL2_1.log" 2>&1 &
sleep 2