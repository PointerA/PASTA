import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import tensorflow as tf
# import matplotlib.pyplot as plt

# from tqdm import tqdm
import numpy as np
import random
import time
import copy
import threading

# from models.vision import resnet18, MLP2
# from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, multiclass_auc
# from utils.communication_protocol_funcs import get_size_of

# from evaluates.attacks.attack_api import apply_attack
# from evaluates.defenses.defense_api import apply_defense
# from evaluates.defenses.defense_functions import *
# from utils.constants import *
# import utils.constants as shared_var
# from utils.marvell_functions import KL_gradient_perturb
# from utils.noisy_label_functions import add_noise
# from utils.noisy_sample_functions import noisy_sample
# from utils.communication_protocol_funcs import compress_pred,Cache,ins_weight
# from evaluates.attacks.attack_api import AttackerLoader

from utils.utils import send_msg

# tf.compat.v1.enable_eager_execution() 

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40,'diabetes':0.69,\
'nuswide': 0.88, 'breast_cancer_diagnose':0.88,'adult_income':0.84,'cora':0.72,\
'avazu':0.83,'criteo':0.74,'nursery':0.99,'credit':0.82}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL_asyn(object):

    def __init__(self, args):
        self.args = args
        self.client_id = args.client_id
        self.k = args.k
        self.device = args.device
        self.dataset_name = args.dataset
        # self.train_dataset = args.train_dst
        # self.val_dataset = args.test_dst
        # self.half_dim = args.half_dim
        self.epochs = args.main_epochs
        self.lr = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        # self.num_classes = args.num_classes
        # self.num_class_list = args.num_class_list
        self.num_classes = args.num_classes
        self.exp_res_dir = f'exp_result/{args.dataset}_{args.seed}/{str(args.communication_protocol)}/'

        self.parties = args.parties
        
        self.Q = args.Q # FedBCD

        self.parties_data = None
        self.gt_one_hot_label = None
        self.clean_one_hot_label  = None
        self.pred_list = []
        self.pred_list_clone = []
        self.pred_gradients_list = []
        self.pred_gradients_list_clone = []
        
        # FedBCD related
        self.local_pred_list = []
        self.local_pred_list_clone = []
        self.local_pred_gradients_list = []
        self.local_pred_gradients_list_clone = []
        
        self.loss = None
        self.train_acc = None
        self.flag = 1
        self.test_acc = 0
        self.stopping_iter = 0
        self.stopping_time = 0.0
        self.stopping_commu_cost = 0
        self.communication_cost = 0


        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        self.final_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

        self.num_update_per_batch = args.num_update_per_batch
        self.num_batch_per_workset = args.Q #args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch*self.num_batch_per_workset
        
        self.total_batch = 1
        self.start_time = None
        self.max_staleness = args.Q
        self.staleness = args.Q
        self.Control_signal = [0 for _ in range(self.k)]
        self.num_batch_to_send_each_round = [1]


    
    def pred_transmit(self, model = None): # Active party gets pred from passive parties
        pred, pred_detach = self.parties[self.client_id].give_pred(model)

        if self.client_id == (self.k-1): # Active party update local pred
            pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
            self.parties[self.client_id].update_local_pred(pred_clone)
        
        if self.client_id < (self.k-1): # Passive party sends pred for aggregation
            if self.args.communication_protocol in ['FlexVFL']:
                each_local_training_time = 0
                if self.parties[self.client_id].local_training_count > 0:
                    each_local_training_time = self.parties[self.client_id].total_local_training_time / self.parties[self.client_id].local_training_count
                send_msg(self.args.socket, (self.client_id, pred_detach, self.num_total_comms, each_local_training_time))
            else:
                send_msg(self.args.socket, (self.client_id, pred_detach, self.num_total_comms))
    
    def gradient_transmit(self):  # Active party sends gradient to passive parties
        if self.args.communication_protocol in ['ours']:
            self.parties[self.k-1].generate_Control_signal(self.Control_signal)

        Qs = [self.args.Q for i in range(self.k)]
        #update Q
        if self.args.communication_protocol in ['FlexVFL']:
            if self.num_total_comms >= 1:
                each_local_training_time = 0
                if self.parties[self.client_id].local_training_count > 0:
                    each_local_training_time = self.parties[self.client_id].total_local_training_time / self.parties[self.client_id].local_training_count
                self.parties[self.client_id].each_party_local_training_time[self.k-1] = each_local_training_time

                max_time_per_batch = min(self.args.Q * self.parties[self.client_id].each_party_local_training_time[i] + 2 * self.parties[self.client_id].each_party_communication_time[i] for i in range(self.k))
                floor_time_per_client = [self.parties[self.client_id].each_party_local_training_time[i] + 2 * self.parties[self.client_id].each_party_communication_time[i] for i in range(self.k)]
                max_time_per_batch = max(max(floor_time_per_client), max_time_per_batch)
                Qs = [1 for i in range(self.k)]
                for i in range(self.k):
                    temp = floor_time_per_client[i]
                    while(temp < max_time_per_batch and Qs[i] < self.args.Q):
                        temp += self.parties[self.client_id].each_party_local_training_time[i]
                        if temp <= max_time_per_batch:
                            Qs[i] += 1
                self.parties[self.client_id].Q_this_round = Qs[self.client_id]
            
        for ik in range(self.args.k-1):
            with self.args.lock:
                pred_detach = self.parties[self.k-1].message_caches[ik].pop(self.parties[self.k-1].finish_batch+1, None)
            if(pred_detach is None):
                # start_time = time.time()
                pred_detach = self.parties[self.k-1].message_queues[ik].get()
                # end_time = time.time()
                # if(ik == 2):
                #     print("ik == 2, ", end_time-start_time)
                # else:
                #     print(end_time-start_time)
            pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
            self.parties[self.k-1].receive_pred(pred_clone, ik)

        gradient = self.parties[self.k-1].give_gradient() # gradient_clone
        # active party update local gradient
        self.parties[self.k-1].update_local_gradient(gradient[self.k-1])

        def send_gradient(ik):
            if self.args.communication_protocol in ['ours']:
                send_msg(self.args.socket[ik][0], (self.client_id, gradient[ik], self.num_total_comms, self.Control_signal[ik]))
                self.Control_signal[ik] = 0
            elif self.args.communication_protocol in ['FlexVFL']:
                mb = send_msg(self.args.socket[ik][0], (self.client_id, gradient[ik], self.num_total_comms, Qs[ik]))
                self.parties[self.client_id].each_party_communication_time[ik] = mb / self.parties[self.client_id].each_party_communication_power[ik]
            else:
                send_msg(self.args.socket[ik][0], (self.client_id, gradient[ik], self.num_total_comms))
        # active party transfer gradient to passive parties
        threads = []
        for ik_ in range(self.args.k-1):
            t = threading.Thread(target=send_gradient, args=(ik_,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return
    
    def label_to_one_hot(self, target, num_classes=10):
        # print('label_to_one_hot:', target, type(target))
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size())
            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def LR_Decay(self,i_epoch):
        self.parties[self.client_id].LR_decay(i_epoch)
        if self.client_id == self.k-1:
            self.parties[self.client_id].global_LR_decay(i_epoch)

    def Staleness_Decay(self,i_epoch):
        self.staleness = int(self.max_staleness / (np.sqrt(i_epoch + 1)))
        
    def train_batch(self, parties_data, batch_label):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
        '''
        gt_one_hot_label = batch_label
        
        if self.client_id == self.k-1:
            self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
        # allocate data to each party
        self.parties[self.client_id].obtain_local_data(parties_data[0])

        # ====== normal vertical federated learning ======
        torch.autograd.set_detect_anomaly(True)
        # ======== Commu ===========
        if self.args.communication_protocol in ['Vanilla','FedBCD_p','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            for q in range(self.Q):
                if q == 0: 
                    # exchange info between parties
                    self.pred_transmit()
                    if self.client_id == self.k-1:
                        self.gradient_transmit() 
                    # update parameters
                    if self.client_id == self.k-1:
                        self.parties[self.k-1].global_backward()
                        self.parties[self.client_id].local_backward()
                    else:
                        with self.args.lock:
                            self.parties[self.client_id].local_gradient = self.parties[self.client_id].message_caches.pop(self.parties[self.client_id].finish_batch+1, None)
                        if(self.parties[self.client_id].local_gradient is None):
                            self.parties[self.client_id].local_gradient = self.parties[self.client_id].local_gradients.get()
                        
                        self.parties[self.client_id].local_gradient = torch.autograd.Variable(self.parties[self.client_id].local_gradient, requires_grad=True).to(self.args.device)
                        # print(f'training on cuda{torch.cuda.current_device()}')
                        # print(self.parties[self.client_id].local_pred.device)
                        # self.parties[self.client_id].local_gradient.to(self.parties[self.client_id].local_pred.device)
                        # print(self.parties[self.client_id].local_gradient.device)
                        # print(self.parties[self.client_id].local_model.device)
                        self.parties[self.client_id].local_backward()
                    self.parties[self.client_id].finish_batch += 1
                    if (self.parties[self.client_id].finish_batch + 1) % self.total_batch == 0:
                        current_time = time.time()
                        current_epoch = int((self.parties[self.client_id].finish_batch + 1) / self.total_batch)
                        print("epoch ",current_epoch, " takes time", current_time-self.start_time)
                        self.save_trained_models(current_epoch)
                else: # FedBCD: additional iterations without info exchange
                    # for passive party, do local update without info exchange
                    if self.client_id == self.k-1:
                        _pred, _pred_clone = self.parties[self.k-1].give_pred() 
                        _gradient = self.parties[self.k-1].give_gradient()
                        self.parties[self.k-1].global_backward()
                        self.parties[self.k-1].local_backward()
                    else:
                        _pred, _pred_clone= self.parties[self.client_id].give_pred() 
                        self.parties[self.client_id].local_backward()
        elif self.args.communication_protocol in ['FlexVFL']: # parallel FedBCD & noBCD situation
            for q in range(self.parties[self.client_id].Q_this_round):
                if q == 0: 
                    # exchange info between parties
                    self.pred_transmit()
                    if self.client_id == self.k-1:
                        self.gradient_transmit() 
                    # update parameters
                    if self.client_id == self.k-1:
                        self.parties[self.k-1].global_backward()
                        self.parties[self.client_id].local_backward()
                    else:
                        with self.args.lock:
                            self.parties[self.client_id].local_gradient = self.parties[self.client_id].message_caches.pop(self.parties[self.client_id].finish_batch+1, None)
                        if(self.parties[self.client_id].local_gradient is None):
                            self.parties[self.client_id].local_gradient = self.parties[self.client_id].local_gradients.get()
                        self.parties[self.client_id].local_gradient = torch.autograd.Variable(self.parties[self.client_id].local_gradient, requires_grad=True).to(self.args.device)
                        self.parties[self.client_id].local_backward()
                    self.parties[self.client_id].finish_batch += 1
                    self.parties[self.client_id].num_local_train.append(0)
                    if (self.parties[self.client_id].finish_batch + 1) % self.total_batch == 0:
                        current_time = time.time()
                        current_epoch = int((self.parties[self.client_id].finish_batch + 1) / self.total_batch)
                        print("epoch ",current_epoch, " takes time", current_time-self.start_time)
                        self.save_trained_models(current_epoch)   
                else:
                    start_time = time.time()
                    if self.client_id == self.k-1:
                        _pred, _pred_clone = self.parties[self.k-1].give_pred() 
                        _gradient = self.parties[self.k-1].give_gradient()
                        self.parties[self.k-1].global_backward()
                        self.parties[self.k-1].local_backward()
                        end_time = time.time()
                        self.parties[self.client_id].total_local_training_time += end_time - start_time
                        self.parties[self.client_id].local_training_count += 1
                    else:
                        
                        _pred, _pred_clone= self.parties[self.client_id].give_pred() 
                        self.parties[self.client_id].local_backward()
                        end_time = time.time()
                        self.parties[self.client_id].total_local_training_time += end_time - start_time
                        self.parties[self.client_id].local_training_count += 1

                    self.parties[self.client_id].num_local_train[-1] += 1




        elif self.args.communication_protocol in ['ours']:
            
            num_of_batch_to_send = self.parties[self.client_id].num_of_batch_to_send
            if self.num_batch_to_send_each_round[-1] > num_of_batch_to_send:
                num_of_batch_to_send = self.num_batch_to_send_each_round[-1] - 1
            elif self.num_batch_to_send_each_round[-1] < num_of_batch_to_send:
                num_of_batch_to_send = self.num_batch_to_send_each_round[-1] + 1
            self.num_batch_to_send_each_round.append(num_of_batch_to_send)
            self.parties[self.client_id].num_of_batch_to_send = num_of_batch_to_send
            
            Q = max(1,int(self.staleness / num_of_batch_to_send + 0.5))
            if self.client_id == self.k-1:
                for q in range(Q):
                    if q == 0: 
                        # exchange info between parties
                        self.pred_transmit()
                        self.gradient_transmit()
                        # update parameters
                        self.parties[self.k-1].global_backward()
                        self.parties[self.client_id].local_backward()
                        self.parties[self.client_id].finish_batch += 1
                        self.parties[self.client_id].num_local_train.append(0)
                        if (self.parties[self.client_id].finish_batch + 1) % self.total_batch == 0:
                            current_time = time.time()
                            current_epoch = int((self.parties[self.client_id].finish_batch + 1) / self.total_batch)
                            print("epoch ",current_epoch, " takes time", current_time-self.start_time)
                            self.save_trained_models(current_epoch)
                    else: 
                        if(self.parties[self.client_id].check_ready()):
                            break
                        _pred, _pred_clone = self.parties[self.k-1].give_pred() 
                        _gradient = self.parties[self.k-1].give_gradient()
                        self.parties[self.k-1].global_backward()
                        self.parties[self.k-1].local_backward()
                        self.parties[self.client_id].num_local_train[-1] += 1

            else:
                while(self.parties[self.client_id].local_preds.qsize() > num_of_batch_to_send):
                    #backward
                    model_clone = self.parties[self.client_id].local_models.get()
                    self.parties[self.client_id].local_batch_data = self.parties[self.client_id].local_datas.get()
                    self.parties[self.client_id].local_pred = self.parties[self.client_id].local_preds.get()
                    with self.args.lock:
                        self.parties[self.client_id].local_gradient = self.parties[self.client_id].message_caches.pop(self.parties[self.client_id].finish_batch+1, None)
                    if(self.parties[self.client_id].local_gradient is None):
                        self.parties[self.client_id].local_gradient = self.parties[self.client_id].local_gradients.get()
                    self.parties[self.client_id].local_gradient = torch.autograd.Variable(self.parties[self.client_id].local_gradient, requires_grad=True).to(self.args.device)
                    self.parties[self.client_id].local_backward(model_clone)
                    self.parties[self.client_id].finish_batch += 1
                    self.parties[self.client_id].num_local_train.append(0)
                    if (self.parties[self.client_id].finish_batch + 1) % self.total_batch == 0:
                        current_time = time.time()
                        current_epoch = int((self.parties[self.client_id].finish_batch + 1) / self.total_batch)
                        print("epoch ",current_epoch, " takes time", current_time-self.start_time)
                        self.save_trained_models(current_epoch)

                    #local training
                    while(self.parties[self.client_id].num_local_train[-1] < Q-1):
                        if(self.parties[self.client_id].check_ready()):
                            break
                        _pred, _pred_clone= self.parties[self.client_id].give_pred() 
                        self.parties[self.client_id].local_backward()
                        self.parties[self.client_id].num_local_train[-1] += 1

                #forward
                self.parties[self.client_id].obtain_local_data(parties_data[0])
                model_clone = copy.deepcopy(self.parties[self.client_id].local_model)
                self.parties[self.client_id].local_models.put(model_clone)
                self.pred_transmit(model_clone)
                self.parties[self.client_id].local_datas.put(self.parties[self.client_id].local_batch_data)
                self.parties[self.client_id].local_preds.put(self.parties[self.client_id].local_pred)
                

                while(self.parties[self.client_id].local_preds.qsize() > num_of_batch_to_send):
                    #backward
                    model_clone = self.parties[self.client_id].local_models.get()
                    self.parties[self.client_id].local_batch_data = self.parties[self.client_id].local_datas.get()
                    self.parties[self.client_id].local_pred = self.parties[self.client_id].local_preds.get()
                    with self.args.lock:
                        self.parties[self.client_id].local_gradient = self.parties[self.client_id].message_caches.pop(self.parties[self.client_id].finish_batch+1, None)
                    if(self.parties[self.client_id].local_gradient is None):
                        self.parties[self.client_id].local_gradient = self.parties[self.client_id].local_gradients.get()
                    self.parties[self.client_id].local_gradient = torch.autograd.Variable(self.parties[self.client_id].local_gradient, requires_grad=True).to(self.args.device)
                    self.parties[self.client_id].local_backward(model_clone)
                    self.parties[self.client_id].finish_batch += 1
                    self.parties[self.client_id].num_local_train.append(0)
                    if (self.parties[self.client_id].finish_batch + 1) % self.total_batch == 0:
                        current_time = time.time()
                        current_epoch = int((self.parties[self.client_id].finish_batch + 1) / self.total_batch)
                        print("epoch ",current_epoch, " takes time", current_time-self.start_time)
                        self.save_trained_models(current_epoch)

                    #local training
                    while(self.parties[self.client_id].num_local_train[-1] < Q-1):
                        if(self.parties[self.client_id].check_ready()):
                            break
                        _pred, _pred_clone= self.parties[self.client_id].give_pred() 
                        self.parties[self.client_id].local_backward()
                        self.parties[self.client_id].num_local_train[-1] += 1
                    
                if int((self.num_total_comms + 1) / self.total_batch) == self.epochs:
                    while(self.parties[self.client_id].finish_batch < self.num_total_comms):
                        #backward
                        model_clone = self.parties[self.client_id].local_models.get()
                        self.parties[self.client_id].local_batch_data = self.parties[self.client_id].local_datas.get()
                        self.parties[self.client_id].local_pred = self.parties[self.client_id].local_preds.get()
                        with self.args.lock:
                            self.parties[self.client_id].local_gradient = self.parties[self.client_id].message_caches.pop(self.parties[self.client_id].finish_batch+1, None)
                        if(self.parties[self.client_id].local_gradient is None):
                            self.parties[self.client_id].local_gradient = self.parties[self.client_id].local_gradients.get()
                        self.parties[self.client_id].local_gradient = torch.autograd.Variable(self.parties[self.client_id].local_gradient, requires_grad=True).to(self.args.device)
                        self.parties[self.client_id].local_backward(model_clone)
                        self.parties[self.client_id].finish_batch += 1
                        self.parties[self.client_id].num_local_train.append(0)
                        if (self.parties[self.client_id].finish_batch + 1) % self.total_batch == 0:
                            current_time = time.time()
                            current_epoch = int((self.parties[self.client_id].finish_batch + 1) / self.total_batch)
                            print("epoch ",current_epoch, " takes time", current_time-self.start_time)
                            self.save_trained_models(current_epoch)

                
                
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================

        if self.client_id == self.k-1:
            real_batch_label = batch_label

            pred = self.parties[self.k-1].global_pred
            loss = self.parties[self.k-1].global_loss
            predict_prob = F.softmax(pred, dim=-1)

            suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(real_batch_label, dim=-1)).item()
            train_acc = suc_cnt / predict_prob.shape[0]
            
            return loss.item(), train_acc
        else:
            return 0, 0

    def train(self):
        if self.args.device == 'cuda':
            cuda_id = self.args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'training on cuda{torch.cuda.current_device()}')
        else:
            print('training on cpu')
        
        self.parties[self.client_id].prepare_data_loader(batch_size=self.batch_size)
        print_every = 1
        self.total_batch = len(self.parties[self.client_id].train_loader)

        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0
        LR_passive_list = []
        LR_active_list = []

        self.num_total_comms = 0
        total_time = 0.0
        flag = 0
        self.current_epoch = 0

        train_loss_i_epoch = 0.0
        train_acc_i_epoch = 0.0
        
        self.start_time = time.time()
        for i_epoch in range(self.epochs):

            self.current_epoch = i_epoch
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            
            data_loader_list = self.parties[self.client_id].train_loader
            train_loss_i_epoch = 0
            train_acc_i_epoch = 0

            self.current_step = 0
            for parties_data in data_loader_list:
                #print(self.num_total_comms)
                if self.client_id == self.k-1:
                    self.gt_one_hot_label = self.label_to_one_hot(parties_data[1], self.num_classes)
                    self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                
                self.parties_data = parties_data

                self.parties[self.client_id].local_model.train()
                if self.client_id == self.k-1:
                    self.parties[self.k-1].global_model.train()
                
                # ====== train batch (start) ======

                self.loss, self.train_acc = self.train_batch(self.parties_data,self.gt_one_hot_label)
                train_loss_i_epoch += self.loss
                train_acc_i_epoch += self.train_acc
                self.num_total_comms = self.num_total_comms + 1
                #print('num_total_comms: ', self.num_total_comms)
                
                # ====== train batch (end) ======

                self.current_step = self.current_step + 1

            # LR decay
            self.LR_Decay(i_epoch)
            self.Staleness_Decay(i_epoch)
            
            # validation
            if i_epoch % print_every == 0:
                # self.save_trained_models(current_epoch)
                print('Epoch {}% \t train_loss:{:.4f} train_acc:{:.4f}'.format(
                        i_epoch, train_loss_i_epoch / self.total_batch, train_acc_i_epoch / self.total_batch))
                train_loss_i_epoch = 0.0
                train_acc_i_epoch = 0.0

        print('num_local_train: ', self.parties[self.client_id].num_local_train)
        if self.client_id < self.k-1:
            print('num_batch_to_send_each_round: ', self.num_batch_to_send_each_round)
        exp_result = f"K|bs|LR|num_class|Q|top_trainable|epoch|communication|main_task_acc,%d|%d|%lf|%d|%d|%d|%d|{self.args.communication_protocol}|{self.test_acc}" %\
            (self.args.k,self.args.batch_size, self.args.main_lr, self.args.num_classes, self.args.Q, self.args.apply_trainable_layer,self.args.main_epochs)
        print(exp_result)
        return self.test_acc,self.stopping_iter,self.stopping_time,self.stopping_commu_cost


    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                # type(model) = <class 'xxxx.ModelName'>
                "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in range(self.args.k)]+[str(type(self.parties[self.args.k-1].global_model)).split('.')[-1].split('\'')[-2]]
            
            }
        else:
            return {
                # "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
                "data": copy.deepcopy(self.parties_data), 
                "label": copy.deepcopy(self.gt_one_hot_label),
                "predict": [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)],
                "gradient": [copy.deepcopy(self.parties[ik].local_gradient) for ik in range(self.k)],
                "local_model_gradient": [copy.deepcopy(self.parties[ik].weights_grad_a) for ik in range(self.k)],
                "train_acc": copy.deepcopy(self.train_acc),
                "loss": copy.deepcopy(self.loss),
                "global_pred":self.parties[self.k-1].global_pred,
                "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "final_global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                
            }

    def save_party_data(self):
        return {
            "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
            "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],
            "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
            "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],
            "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
            "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
            "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],
            "aux_loader": [copy.deepcopy(self.parties[ik].aux_loader) for ik in range(self.k)],
            "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
            "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],
            "batchsize": self.args.batch_size,
            "num_classes": self.args.num_classes
        }
        
        
    def save_trained_models(self, current_epoch):
        dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}/party{self.client_id}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = dir_path + f'epoch{current_epoch}.pkl'
        torch.save(self.parties[self.client_id].local_model.state_dict(), file_path)
        #print("save path:", file_path)
        
        if self.client_id == self.k-1:
            file_path = dir_path + f'global_epoch{current_epoch}.pkl'
            torch.save(self.parties[self.client_id].global_model.state_dict(), file_path)

    # def evaluate_attack(self):
    #     self.attacker = AttackerLoader(self, self.args)
    #     if self.attacker != None:
    #         attack_acc = self.attacker.attack()
    #     return attack_acc

    # def launch_defense(self, gradients_list, _type):
        
    #     if _type == 'gradients':
    #         return apply_defense(self.args, _type, gradients_list)
    #     elif _type == 'pred':
    #         return apply_defense(self.args, _type, gradients_list)
    #     else:
    #         # further extention
    #         return gradients_list

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total
