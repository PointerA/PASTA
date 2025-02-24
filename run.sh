#!/bin/bash

conda_env="iwqos"



source activate $conda_env
cd ~/iwqos/



nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 7 >> "./log/cifar10/fedsgd7_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 0 >> "./log/cifar10/fedsgd0_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 1 >> "./log/cifar10/fedsgd1_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 2 >> "./log/cifar10/fedsgd2_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 3 >> "./log/cifar10/fedsgd3_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 4 >> "./log/cifar10/fedsgd4_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 5 >> "./log/cifar10/fedsgd5_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/fedsgd/8_party/cifar10.json --client_id 6 >> "./log/cifar10/fedsgd6_1.log" 2>&1 &

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 7 >> "./log/cifar10/our7_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 0 >> "./log/cifar10/our0_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 1 >> "./log/cifar10/our1_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 2 >> "./log/cifar10/our2_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 3 >> "./log/cifar10/our3_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 4 >> "./log/cifar10/our4_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 5 >> "./log/cifar10/our5_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/ours3/8_party/cifar10.json --client_id 6 >> "./log/cifar10/our6_1.log" 2>&1 &

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FedBCD7_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FedBCD0_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FedBCD1_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FedBCD2_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FedBCD3_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FedBCD4_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FedBCD5_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FedBCD/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FedBCD6_1.log" 2>&1 &

nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 7 >> "./log/cifar10/FlexVFL7_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 0 >> "./log/cifar10/FlexVFL0_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 1 >> "./log/cifar10/FlexVFL1_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 2 >> "./log/cifar10/FlexVFL2_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 3 >> "./log/cifar10/FlexVFL3_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 4 >> "./log/cifar10/FlexVFL4_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 5 >> "./log/cifar10/FlexVFL5_1.log" 2>&1 &
nohup python -u main_pipeline_asyn.py --seed 97 --gpu 0 --configs ./configs/communication/FlexVFL/8_party/cifar10.json --client_id 6 >> "./log/cifar10/FlexVFL6_1.log" 2>&1 &