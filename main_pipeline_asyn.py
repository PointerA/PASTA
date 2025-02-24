import os
import sys
import numpy as np
import socket
import threading
import time

import random
import logging
import argparse
import torch
#import tensorflow as tf
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import datasets
# import torch.utils
# import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

from load.LoadConfigs import * #load_configs
from load.LoadParty import load_parties
from evaluates.MainTaskVFL_asyn import *
# from evaluates.MainTaskVFLwithBackdoor import *
# from evaluates.MainTaskVFLwithNoisySample import *
# from utils.basic_functions import append_exp_res
import warnings
warnings.filterwarnings("ignore")


from utils.utils import *


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def active_receive_messages(args, conn, addr):
    print(f"Connected with {addr}")
    while True:
        try:
            # 接收消息
            data, mb = recv_msg(conn)
            
            # 将消息放入队列
            with args.lock:
                if args.parties[args.client_id].finish_batch + 1 == data[2]:
                    args.parties[args.client_id].message_queues[data[0]].put(data[1])
                elif args.parties[args.client_id].finish_batch + 1 < data[2]:
                    args.parties[args.client_id].message_caches[data[0]][data[2]] = data[1]
                if args.communication_protocol in ['FlexVFL']:
                    args.parties[args.client_id].each_party_local_training_time[data[0]] = data[3]
                    args.parties[args.client_id].each_party_communication_time[data[0]] += mb / args.parties[args.client_id].each_party_communication_power[data[0]]
        except socket.error as e:
            print(f"Socket error occurred: {e}")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    
    conn.close()
    print(f"Connection with {addr} closed")
    
def passive_receive_messages(args, conn, addr):
    print(f"Connected with {addr}")
    while True:
        try:
            # 接收消息
            data, _ = recv_msg(conn)
            
            # 将消息放入队列
            with args.lock:
                if args.parties[args.client_id].finish_batch + 1 == data[2]:
                    args.parties[args.client_id].local_gradients.put(data[1])
                elif args.parties[args.client_id].finish_batch + 1 < data[2]:
                    args.parties[args.client_id].message_caches[data[2]] = data[1]
                if args.communication_protocol in ['ours']:
                    if data[3] > 0:
                        if args.parties[args.client_id].num_local_train[-1] != 0:
                            args.parties[args.client_id].num_of_batch_to_send = min(args.parties[args.client_id].num_of_batch_to_send + 1, args.parties[args.client_id].maximun_of_batch_to_send)
                    elif data[3] < 0:
                        args.parties[args.client_id].num_of_batch_to_send = max(1, args.parties[args.client_id].num_of_batch_to_send - 1)
                elif args.communication_protocol in ['FlexVFL']:
                    args.parties[args.client_id].Q_this_round = data[3]

        except socket.error as e:
            print(f"Socket error occurred: {e}")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        
    
    conn.close()
    print(f"Connection with {addr} closed")

def evaluate_no_attack(args):
    # No Attack
    set_seed(args.current_seed)
    
    args.lock = threading.Lock()
    threads = []
    
    if args.client_id == args.k-1:
        listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listening_sock.bind((SERVER_ADDR, SERVER_PORT))
        listening_sock.settimeout(TIME_OUT)
        client_sock_all=[]

        # Establish connections to each client, up to n_nodes clients
        while len(client_sock_all) < args.k-1:
            listening_sock.listen(20)
            print("Waiting for incoming connections...")
            (client_sock, (ip, port)) = listening_sock.accept()
            print('Got connection from ', (ip,port))
            client_sock_all.append((client_sock, (ip, port)))
        
        for i in range(args.k-1):
            receive_thread = threading.Thread(target=active_receive_messages, args=(args, client_sock_all[i][0], client_sock_all[i][1]))
            receive_thread.start()
            threads.append(receive_thread)
            
        args.socket = client_sock_all
        
    else:
        sock = socket.socket()
        if args.k == 8:
            sock.bind(CLIENT_ADDRS[args.client_id])
        sock.connect((SERVER_ADDR, SERVER_PORT))
        sock.settimeout(TIME_OUT)
        receive_thread = threading.Thread(target=passive_receive_messages, args=(args, sock, (SERVER_ADDR, SERVER_PORT)))
        receive_thread.start()
        threads.append(receive_thread)
        
        args.socket = sock

    vfl = MainTaskVFL_asyn(args)
    
    #timer_function(10)
    compute_thread = None
    if args.dataset not in ['cora']:
        #main_acc , stopping_iter, stopping_time, stopping_commu_cost= vfl.train()
        compute_thread = threading.Thread(target=vfl.train)
        compute_thread.start()
    else:
        #main_acc, stopping_iter, stopping_time = vfl.train_graph()
        compute_thread = threading.Thread(target=vfl.train_graph)
        compute_thread.start()
    compute_thread.join()

    for t in threads:
        t.join()
    main_acc_noattack = 0
    
    return vfl, main_acc_noattack

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--client_id', type=int, default=0, help='0~k-2 is passive, k-1 is active')
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--configs', type=str, default='test', help='configure json file path')
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save the trained model')
    args = parser.parse_args()

    # for seed in range(97,102): # test 5 times 
    # for seed in [60]:
    # for seed in [97,98,99,100,101]: # test 5 times 
    for seed in [args.seed]: # test 5 times 
        args.current_seed = seed
        set_seed(seed)
        print('================= iter seed ',seed,' =================')
        
        args = load_basic_configs(args.configs, args)
        args.need_auxiliary = 0 # no auxiliary dataset for attackerB

        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')
        else:
            print('running on cpu')

        
        ####### load configs from *.json files #######
        ############ Basic Configs ############
        
        # for mode in [0]:
            
        #     if mode == 0:
        #         args.global_model = 'ClassificationModelHostHead'
        #     else:
        #         args.global_model = 'ClassificationModelHostTrainableHead'
        #     args.apply_trainable_layer = mode

        mode = args.apply_trainable_layer 
        print('============ apply_trainable_layer=',args.apply_trainable_layer,'============')
        #print('================================')
    
        assert args.dataset_split != None, "dataset_split attribute not found config json file"
        assert 'dataset_name' in args.dataset_split, 'dataset not specified, please add the name of the dataset in config json file'
        args.dataset = args.dataset_split['dataset_name']
        # print(args.dataset)  

        args.basic_vfl_withaux = None
        args.main_acc_noattack_withaux = None
        args.basic_vfl = None
        args.main_acc_noattack = None

        args = load_parties(args)

        args.basic_vfl, args.main_acc_noattack = evaluate_no_attack(args)
        
        







