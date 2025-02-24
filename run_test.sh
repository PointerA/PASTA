# nohup python -u main_pipeline_test.py --seed 97 --configs ./configs/communication/fedsgd/8_party/cifar10.json >> "../result/fedsgd.log" 2>&1 &
# sleep 600
# nohup python -u main_pipeline_test.py --seed 98 --configs ./configs/communication/fedsgd/8_party/cifar10.json >> "../result/fedsgd.log" 2>&1 &
# sleep 400
# nohup python -u main_pipeline_test.py --seed 99 --configs ./configs/communication/fedsgd/8_party/cifar10.json >> "../result/fedsgd.log" 2>&1 &

# sleep 400
# nohup python -u main_pipeline_test.py --seed 97 --configs ./configs/communication/ours3/8_party/cifar10.json >> "../result/ours.log" 2>&1 &
# sleep 400
# nohup python -u main_pipeline_test.py --seed 98 --configs ./configs/communication/ours3/8_party/cifar10.json >> "../result/ours.log" 2>&1 &
nohup python -u main_pipeline_test.py --seed 99 --configs ./configs/communication/ours3/8_party/cifar10.json >> "../result/ours.log" 2>&1 &

sleep 400
nohup python -u main_pipeline_test.py --seed 97 --configs ./configs/communication/FedBCD/8_party/cifar10.json >> "../result/FedBCD.log" 2>&1 &
sleep 400
nohup python -u main_pipeline_test.py --seed 98 --configs ./configs/communication/FedBCD/8_party/cifar10.json >> "../result/FedBCD.log" 2>&1 &
sleep 400
nohup python -u main_pipeline_test.py --seed 99 --configs ./configs/communication/FedBCD/8_party/cifar10.json >> "../result/FedBCD.log" 2>&1 &

sleep 400
nohup python -u main_pipeline_test.py --seed 97 --configs ./configs/communication/FlexVFL/8_party/cifar10.json >> "../result/FlexVFL.log" 2>&1 &
sleep 400
nohup python -u main_pipeline_test.py --seed 98 --configs ./configs/communication/FlexVFL/8_party/cifar10.json >> "../result/FlexVFL.log" 2>&1 &
sleep 400
nohup python -u main_pipeline_test.py --seed 99 --configs ./configs/communication/FlexVFL/8_party/cifar10.json >> "../result/FlexVFL.log" 2>&1 &