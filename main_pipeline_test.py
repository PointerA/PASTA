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
# import tensorflow as tf
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import datasets
# import torch.utils
# import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

from load.LoadConfigs import * #load_configs
from load.LoadParty import test_load_parties
from evaluates.MainTaskVFL_asyn import *
# from evaluates.MainTaskVFLwithBackdoor import *
# from evaluates.MainTaskVFLwithNoisySample import *
# from utils.basic_functions import append_exp_res
import warnings
warnings.filterwarnings("ignore")
from utils.basic_functions import multiclass_auc


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

def label_to_one_hot(target, num_classes=10, args=None):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32).to(args.device)
    except:
        target = torch.unsqueeze(target, 1).to(args.device)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes, device=args.device)
        onehot_target.scatter_(1, target, 1)
    return onehot_target

def load_model(args, i_epoch):
    exp_res_dir = f'./exp_result/{args.dataset}_{args.seed}'+f'/{str(args.communication_protocol)}/'
    for ik in range(args.k):
        dir_path = exp_res_dir + f'trained_models/parties{args.k}_topmodel{args.apply_trainable_layer}/party{ik}/'
        model_path = dir_path + f'epoch{i_epoch}.pkl'
        args.parties[ik].local_model.load_state_dict(torch.load(model_path))
        
        if ik == args.k-1:
            model_path = dir_path + f'global_epoch{i_epoch}.pkl'
            args.parties[ik].global_model.load_state_dict(torch.load(model_path))

def test_no_attack(args):
    for ik in range(args.k):
        args.parties[ik].prepare_data_loader(batch_size=args.batch_size)
        
    #print(args)
    
    for i_epoch in range(1, args.main_epochs+1):
        postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
        
        load_model(args, i_epoch)
        for ik in range(args.k):
            args.parties[ik].local_model.eval()
        args.parties[args.k-1].global_model.eval()
        
        suc_cnt = 0
        sample_cnt = 0
        noise_suc_cnt = 0
        noise_sample_cnt = 0
        test_preds = []
        test_targets = []
        with torch.no_grad():
            data_loader_list = [args.parties[ik].test_loader for ik in range(args.k)]
            print(len(args.parties[0].train_loader))
            for parties_data in zip(*data_loader_list):
                # print("test", parties_data[0][0].size(),parties_data[args.k-1][0].size(),parties_data[args.k-1][1].size())

                gt_val_one_hot_label = label_to_one_hot(parties_data[args.k-1][1], args.num_classes, args)
                gt_val_one_hot_label = gt_val_one_hot_label.to(args.device)

                pred_list = []
                for ik in range(args.k):
                    _local_pred = args.parties[ik].local_model(parties_data[ik][0])

                    pred_list.append(_local_pred)

                # Normal Evaluation
                test_logit, test_loss = args.parties[args.k-1].aggregate(pred_list, gt_val_one_hot_label, test="True")
                enc_predict_prob = F.softmax(test_logit, dim=-1)
                
                test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                predict_label = torch.argmax(enc_predict_prob, dim=-1)

                actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                sample_cnt += predict_label.shape[0]
                suc_cnt += torch.sum(predict_label == actual_label).item()
                test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))

            test_acc = suc_cnt / float(sample_cnt)
            test_preds = np.vstack(test_preds)
            test_targets = np.vstack(test_targets)
            test_auc = np.mean(multiclass_auc(test_targets, test_preds))
            postfix['test_acc'] = '{:.2f}%'.format(test_acc * 100)
            postfix['test_auc'] = '{:.2f}%'.format(test_auc * 100)
            
            print('Epoch {}% \t test_acc:{:.4f} test_auc:{:.4f}'.format(
                i_epoch, test_acc, test_auc))

    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--client_id', type=int, default=0, help='0~k-2 is passive, k-1 is active')
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--configs', type=str, default='/home/users/zhangxx/wt/test/asyn/asynVFL/configs/communication/fedsgd/4_party/cifar10.json', help='configure json file path')
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

        args = test_load_parties(args)
    

        test_no_attack(args)




