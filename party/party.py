import os
import sys
import numpy as np
import random
import time
import copy
sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader

#from evaluates.attacks.attack_api import AttackerLoader
#from evaluates.defenses.defense_api import DefenderLoader
from load.LoadDataset import load_dataset_per_party, load_dataset_per_party_backdoor,load_dataset_per_party_noisysample
from load.LoadModels import load_models_per_party

#from utils.noisy_label_functions import add_noise
#from utils.noisy_sample_functions import noisy_sample
#from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor,pairwise_dist
from utils.communication_protocol_funcs import Cache

from sys import getsizeof

class Party(object):
    def __init__(self, args, index):
        self.name = "party#" + str(index + 1)
        self.index = index
        self.args = args
        # data for training and testing
        self.half_dim = -1
        self.train_data = None
        self.test_data = None
        self.aux_data = None
        self.train_label = None
        self.test_label = None
        self.aux_label = None
        self.train_attribute = None
        self.test_attribute = None
        self.aux_attribute = None
        self.train_dst = None
        self.test_dst = None
        self.aux_dst = None
        self.train_loader = None
        self.test_loader = None
        self.aux_loader = None
        self.attribute_loader = None
        self.attribute_iter = None
        self.local_batch_data = None
        # backdoor poison data and label and target images list
        self.train_poison_data = None
        self.train_poison_label = None
        self.test_poison_data = None
        self.test_poison_label = None
        self.train_target_list = None
        self.test_target_list = None
        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None

        # attack and defense
        # self.attacker = None
        self.defender = None

        self.prepare_data(args, index)
        self.prepare_model(args, index)
        # self.prepare_attacker(args, index)
        # self.prepare_defender(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None

        self.cache = Cache()
        self.prev_batches = []
        self.num_local_updates = 0
        
        #FlexVFL
        self.total_local_training_time = 0.
        self.local_training_count = 0
        self.Q_this_round = args.Q

        #asyn
        self.finish_batch = -1 # self.finish_batch != num_total_comms
        self.num_of_batch_to_send = 1
        self.maximun_of_batch_to_send = args.maximun_of_batch_to_send
        self.num_local_train = [0]

    def receive_gradient(self, gradient):
        self.local_gradient = gradient
        return

    def give_pred(self, model=None):
        # start_time = time.time()
        if model is None:
            self.local_pred = self.local_model(self.local_batch_data)
        else:
            #ours
            flag = 'resnet' in self.args.model_list[str(self.index)]['type'].lower() or 'cnn' in self.args.model_list[str(self.index)]['type'].lower()
            flag = flag or 'vgg' in self.args.model_list[str(self.index)]['type'].lower()

            #for BN layers
            # if(flag):
            #     momentum = 0.1
            #     para_dict = model.state_dict()
            #     temp = {}
            #     for name, param in para_dict.items():
            #         if 'running_mean' in name.lower() or 'running_var' in name.lower() or 'num_batches_tracked' in name.lower():
            #             temp[name] = copy.deepcopy(param)
                    
            self.local_pred = model(self.local_batch_data)
            
            #for BN layers
            # if(flag):
            #     para_dict = self.local_model.state_dict()
            #     for name, param in para_dict.items():
            #         if 'num_batches_tracked' in name.lower():
            #             if torch.all(model.state_dict()[name] - temp[name] == 1) == False:
            #                 print("BN error")
            #             param += model.state_dict()[name] - temp[name]
            #         elif 'running_mean' in name.lower() or 'running_var' in name.lower():
            #             para_dict[name] = (1 - momentum) * param + model.state_dict()[name] - (1 - momentum) * temp[name]
            #     self.local_model.load_state_dict(para_dict, strict=False)
            if(flag):
                para_dict = model.state_dict()
                temp = {}
                for name, param in para_dict.items():
                    if 'running_mean' in name.lower() or 'running_var' in name.lower() or 'num_batches_tracked' in name.lower():
                        temp[name] = copy.deepcopy(param)
                self.local_model.load_state_dict(temp, strict=False)

        self.local_pred_clone = self.local_pred.detach().clone()
        
        # end_time = time.time()
        # print(end_time-start_time)

        return self.local_pred, self.local_pred_clone

    def prepare_data(self, args, index):
        # prepare raw data for training
        (
            args,
            self.half_dim,
            train_dst,
            test_dst,
        ) = load_dataset_per_party(args, index)
        if len(train_dst) == 2:
            self.train_data, self.train_label = train_dst
            self.test_data, self.test_label = test_dst
        elif len(train_dst) == 3:
            self.train_data, self.train_label, self.train_attribute = train_dst
            self.test_data, self.test_label, self.test_attribute = test_dst

    def prepare_data_loader(self, batch_size):
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size) # , shuffle=True
        self.test_loader = DataLoader(self.test_dst, batch_size=batch_size) # , shuffle=True
        if self.args.need_auxiliary == 1 and self.aux_dst != None:
            self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size)
        if self.train_attribute != None:
            self.attribute_loader = DataLoader(self.train_attribute, batch_size=batch_size)
            self.attribute_iter = iter(self.attribute_loader)

    def prepare_model(self, args, index):
        # prepare model and optimizer
        (
            args,
            self.local_model,
            self.local_model_optimizer,
            self.global_model,
            self.global_model_optimizer,
        ) = load_models_per_party(args, index)

    # def prepare_attacker(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.attacker = AttackerLoader(args, index, self.local_model)

    # def prepare_defender(self, args, index):
    #     if index in args.attack_configs['party']:
    #         self.defender = DefenderLoader(args, index)
    
    def give_current_lr(self):
        return (self.local_model_optimizer.state_dict()['param_groups'][0]['lr'])

    def LR_decay(self,i_epoch):
        eta_0 = self.args.main_lr
        eta_t = eta_0/(np.sqrt(i_epoch+1))
        for param_group in self.local_model_optimizer.param_groups:
            param_group['lr'] = eta_t 
        
        # if i_epoch >= 1:
        #     eta_t = self.args.main_lr*10/(np.sqrt(i_epoch))
        #     for param_group in self.local_model_optimizer.param_groups:
        #         param_group['lr'] = eta_t 
        # else:
        #     eta_0 = self.args.main_lr
        #     for param_group in self.local_model_optimizer.param_groups:
        #         param_group['lr'] = eta_0
            
    def obtain_local_data(self, data):
        self.local_batch_data = data

    def local_forward():
        # args.local_model()
        pass

    # def local_backward(self):
    #     # update local model
    #     self.local_model_optimizer.zero_grad()
    #     # ########## for passive local mid loss (start) ##########
    #     # if passive party in defense party, do
    #     if (
    #         self.args.apply_mid == True
    #         and (self.index in self.args.defense_configs["party"])
    #         and (self.index < self.args.k - 1)
    #         ):
    #         # get grad for local_model.mid_model.parameters()
    #         self.local_model.mid_loss.backward(retain_graph=True)
    #         self.local_model.mid_loss = torch.empty((1, 1)).to(self.args.device)
    #     # ########## for passive local mid loss (end) ##########
    #     self.weights_grad_a = torch.autograd.grad(
    #         self.local_pred,
    #         self.local_model.parameters(),
    #         grad_outputs=self.local_gradient,
    #         retain_graph=True,
    #     )
    #     for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
    #         if w.requires_grad:
    #             w.grad = g.detach()
    #     self.local_model_optimizer.step()


    def local_backward(self,model=None,weight=None):

        self.num_local_updates += 1 # another update
        
        # update local model
        self.local_model_optimizer.zero_grad()

        torch.autograd.set_detect_anomaly(True)
        if weight != None: # CELU
            ins_batch_cached_grad = torch.mul(weight.unsqueeze(1),self.local_gradient)
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.parameters(),
                grad_outputs=ins_batch_cached_grad,
                retain_graph=True
            )
        else:
            if model != None: # ours:
                    self.weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    model.parameters(),
                    grad_outputs=self.local_gradient,
                    retain_graph=True
                    )
            else:
                self.weights_grad_a = torch.autograd.grad(
                    self.local_pred,
                    self.local_model.parameters(),
                    grad_outputs=self.local_gradient,
                    retain_graph=True
                )
        for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
            if w.requires_grad:
                w.grad = g.detach()

        self.local_model_optimizer.step()
