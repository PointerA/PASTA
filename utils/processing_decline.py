#simulate each round training time. with finish control.

import copy
import numpy as np

class Client:
    def __init__(self, client_id, batchs, epochs, k, training_time = (0.0, 0.0), ceiling=4):
        self.client_id = client_id
        self.status = "Idle"  # default is "Idle", and"forward","backward","local trainning","finish"
        self.training_time = training_time
        self.total_epoch = epochs
        self.batchs = batchs
        self.original_ceiling = ceiling
        self.max_k = k
        
        self.ceiling = ceiling
        self.batch_id = 0  #waiting to train
        self.batch_id_done = -1
        self.signal = -1
        self.total_time = 0
        self.k = 1 #the number of micro batch
        self.epoch = 0
        self.waiting_time = 0

        self.num_local_train = []
        self.training_time_each_batch = []
        self.ks = [1]
    
    def start(self):
        # print(self.client_id, " begins to train")
        self.status = "forward"
        for i in range(self.k):
            self.total_time += self.training_time[0]
            
            '''
            forward(batch_id % num_batchs)
            '''
            self.process_communicate(self.batch_id)
            self.batch_id += 1

    def work(self):
        if self.status == "Idle":
            self.start()
        else:
            self.check_signal()
        
    def get_signal(self, signal, k_change):
        self.signal = signal
        
        if k_change > 0:
            if self.num_local_train[-1] != 0:
                self.k = min(self.k + 1, self.max_k)
        elif k_change < 0:
            self.k = max(1, self.k - 1)


    
    def check_signal(self):
        if(self.signal > self.batch_id_done):
            while(self.signal > self.batch_id_done):
                self.process_forward()
        else:
            if self.status == "backward" or self.status == "local trainning" or self.status == "finish":
                self.process_local_trainning()
            elif self.status == "forward":
                self.total_time += 1
            
        return
    
    def process_forward(self):
        if self.status != "finish" and self.status != "forward" and self.k >= self.ks[-1]:
            # print(self.client_id, " begins to forward")

            self.total_time += self.training_time[0]
            self.status = "forward"
            '''
            forward(batch_id % num_batchs)
            '''
            self.process_communicate(self.batch_id)
            self.batch_id += 1
            if self.batch_id / self.batchs == self.total_epoch:
                self.status = "finish"
        
        if self.status != "finish" and self.k > self.ks[-1]:
            # print(self.client_id, " begins to forward")
            self.k = self.ks[-1] + 1

            self.total_time += self.training_time[0]
            self.status = "forward"
            '''
            forward(batch_id % num_batchs)
            '''
            self.process_communicate(self.batch_id)
            self.batch_id += 1
            if self.batch_id / self.batchs == self.total_epoch:
                self.status = "finish"
        
        if self.k < self.ks[-1]:
            self.k = self.ks[-1] - 1
            if self.status == "forward":
                self.ks.append(self.k+1)
            else:
                self.ks.append(self.k)
        else:
            self.ks.append(self.k)
        
        #forward then backward
        self.process_backward()
        return
    
    def process_backward(self):
        # print(self.client_id, " begins to backward")

        self.total_time += self.training_time[1]
        if self.status != "finish":
            self.status = "backward"
        self.batch_id_done += 1
        if self.batch_id_done % self.batchs == self.batchs - 1:
            self.epoch += 1
            self.ceiling_decline()

        self.num_local_train.append(0)
        self.training_time_each_batch.append(self.total_time)
        '''
        backward(batch_id_done % num_batchs)
        '''
        #self.get_signal()
        return
    
    def process_communicate(self, batch):
        # print(self.client_id, " begins to communicate with server")
        global client2server
        global communication_time
        client2server[self.client_id].append((self.total_time + communication_time[self.client_id], batch))
        return
    
    def process_local_trainning(self):
        if self.num_local_train[-1] >= self.ceiling / self.k:
            if self.status != "finish" and self.k >= self.ks[-1]:
                self.status = "forward"
                self.total_time += self.training_time[0]
                '''
                forward(batch_id % num_batchs)
                '''
                self.process_communicate(self.batch_id)
                self.batch_id += 1
                if self.batch_id / self.batchs == self.total_epoch:
                    self.status = "finish"
            else:
                self.total_time += 1
        else:
            # print(self.client_id, " begins to local train")
            self.num_local_train[-1] += 1
            if self.status != "finish":
                self.status = "local trainning"
            self.total_time += self.training_time[0]
            self.total_time += self.training_time[1]
            '''
            forward(batch_id_done % num_batchs)
            backward(batch_id_done % num_batchs)
            '''
            #self.get_signal()
        return
    
    def ceiling_decline(self):
        # temp = (self.original_ceiling+1) * (self.original_ceiling+1 + 1) / 2
        # for i in range(1, self.original_ceiling+1):
        #     if self.epoch >= int(((i+1)*i/2 * self.total_epoch / temp)+0.5):
        #         self.ceiling = self.original_ceiling - i
        #     else:
        #         break
        self.ceiling = int(self.original_ceiling / (np.sqrt(self.epoch + 1)))
        return
    

class Server:
    def __init__(self, client_num, batchs, epochs, k, training_time = (0.0, 0.0, 0.0), ceiling=4):
        self.client_num = client_num
        self.status = "Idle"  # 初始状态为"Idle", 状态有"forward","backward","local trainning","compute","finish"
        self.training_time = training_time
        self.total_epoch = epochs
        self.batchs = batchs
        self.original_ceiling = ceiling
        
        self.ceiling = ceiling
        self.batch_id = 0  # train now
        self.batch_id_done = -1
        self.total_time = 0
        self.waiting_time = 0
        self.epoch = 0

        self.num_local_train = []
        self.training_time_each_batch = []

        self.collections = [-1 for _ in range(self.client_num + 1)]
        self.k_change = [0 for _ in range(self.client_num + 1)]

    
    def start(self):
        # print("server", " begins to train")

        self.status = "forward"
        self.total_time += self.training_time[0]
        '''
        forward(batch_id % num_batchs)
        '''
        self.collections[0] = 0

    
    def work(self):
        if self.status == "Idle":
            self.start()
        elif self.status == "finish":
            return
        else:
            self.check_collection()

    def collect(self, client_id, batch_id):
        self.collections[client_id] = batch_id

    def check_collection(self):
        # to see if finish collecting the batch from all parties.
        flag = True
        for i in range(1, len(self.collections)):
            if self.collections[i] > self.batch_id_done:
                continue
            else:
                flag = False
                break
        if flag:
            if self.status == "forward":
                self.process_compute()
            else:
                self.process_forward()
        else:
            if self.status == "backward" or self.status == "local trainning":
                self.process_local_trainning()
            elif self.status == "forward":
                self.total_time += 1
                if self.batch_id_done != -1 and self.waiting_time == 0:
                    self.waiting_time = 1
                    for i in range(1, len(self.collections)):
                        self.k_change[i] = -(self.collections[i] - self.batch_id_done - 1)


        return
    
    def process_forward(self):
        # print("server", " begins to forward")

        self.total_time += self.training_time[0]
        self.status = "forward"
        '''
        forward(batch_id % num_batchs)
        '''
        self.collections[0] = self.batch_id

        #forward then compute
        self.process_compute()
        return
    
    def process_compute(self):
        # print("server", " begins to compute")

        if self.waiting_time == 0:
            for i in range(1, len(self.collections)):
                self.k_change[i] = -(self.collections[i] - self.batch_id_done - 1)


        self.total_time += self.training_time[2]
        self.status = "compute"
        self.waiting_time = 0
        '''
        compute
        '''
        self.process_communicate()
        #forward then backward
        self.process_backward()
        return
    
    def process_backward(self):
        # print("server", " begins to backward")

        self.total_time += self.training_time[1]
        self.status = "backward"
    
        self.num_local_train.append(0)
        self.training_time_each_batch.append(self.total_time)
        '''
        backward(batch_id % num_batchs)
        '''

        #batch iter
        self.batch_id_done += 1
        self.batch_id += 1
        if self.batch_id_done % self.batchs == self.batchs - 1:
            self.epoch += 1
            self.ceiling_decline()
        if self.epoch == self.total_epoch:
            self.status = "finish"
        #self.check_collection()
        return
    
    def process_communicate(self):
        # print("server", " begins to communicate")

        global server2client
        global communication_time
        for i in range(self.client_num):
            server2client[i+1].append((self.total_time + communication_time[i+1], self.batch_id, self.k_change[i+1]))

        self.k_change = [0 for _ in range(self.client_num + 1)]
        return
    
    def process_local_trainning(self):
        if self.num_local_train[-1] >= self.ceiling:
            if self.status != "finish":
                # print("server", " begins to forward")
                self.total_time += self.training_time[0]
                self.status = "forward"
                '''
                forward(batch_id % num_batchs)
                '''
                self.collections[0] = self.batch_id
            else:
                self.total_time += 1
        else:
            # print("server", " begins to local train")
            self.num_local_train[-1] += 1
            self.status = "local trainning"
            self.total_time += self.training_time[0]
            self.total_time += self.training_time[1]
            '''
            forward(batch_id_done % num_batchs)
            backward(batch_id_done % num_batchs)
            '''
            #self.check_collection()
        return
    
    def ceiling_decline(self):
        # temp = (self.original_ceiling+1) * (self.original_ceiling+1 + 1) / 2
        # for i in range(1, self.original_ceiling+1):
        #     if self.epoch >= int(((i+1)*i/2 * self.total_epoch / temp)+0.5):
        #         self.ceiling = self.original_ceiling - i
        #     else:
        #         break
        self.ceiling = int(self.original_ceiling / (np.sqrt(self.epoch + 1)))
        return

def api4VFLAIR(epochs=30, batchs=98, client_num=4, k=2, ceiling=4):
    worker_list = []
    global communication_time, client2server, server2client
    if client_num == 4:
        worker_list.append(Server(client_num-1, batchs, epochs, k, (10, 20, 10)))
        worker_list.append(Client(1, batchs, epochs, k, (8.0, 15.0)))
        worker_list.append(Client(2, batchs, epochs, k, (25.0, 50.0)))
        worker_list.append(Client(3, batchs, epochs, k, (10.0, 20.0)))
        communication_time = [0.0, 5.0, 5.0, 100.0]
    elif client_num == 2:
        worker_list.append(Server(client_num-1, batchs, epochs, k, (10, 20, 10), ceiling))
        worker_list.append(Client(1, batchs, epochs, k, (10, 20), ceiling))
        communication_time = [0.0, 1000.0]
    
    client2server = [[] for i in range(client_num)]
    server2client = [[] for i in range(client_num)]

    t = 0
    while(client_num*epochs > sum([i.epoch for i in worker_list])):

        for client, queue in enumerate(client2server):
            new_queue = []
            for i in queue:
                if i[0] <= t:
                    worker_list[0].collect(client, i[1])
                else:
                    new_queue.append(i)
            client2server[client] = copy.deepcopy(new_queue)
        
        for client, queue in enumerate(server2client):
            new_queue = []
            for i in queue:
                if i[0] <= t:
                    worker_list[client].get_signal(i[1], i[2])
                else:
                    new_queue.append(i)
            server2client[client] = copy.deepcopy(new_queue)


        for worker in worker_list:
            if t >= worker.total_time:
                worker.work()
        
        t += 1
    
    Qs = []
    for i in range(1, client_num):
        Qs.append(worker_list[i].num_local_train)
    Qs.append(worker_list[0].num_local_train)

    Ks = []
    for i in range(1, client_num):
        Ks.append(worker_list[i].ks)
    
    print(f'comunication time is {communication_time}')
    return Qs,Ks


def api4FlexVFL(epochs=100, batchs=100, client_num=4, k=2, max_local_train=4, floor=0):
    worker_list = []
    global communication_time, client2server, server2client
    if client_num == 4:
        worker_list.append(Server(client_num-1, batchs, epochs, k, (10, 20, 10)))
        worker_list.append(Client(1, batchs, epochs, k, (8.0, 15.0)))
        worker_list.append(Client(2, batchs, epochs, k, (25.0, 50.0)))
        worker_list.append(Client(3, batchs, epochs, k, (10.0, 20.0)))
        communication_time = [0.0, 5.0, 5.0, 100.0]
    elif client_num == 2:
        worker_list.append(Server(client_num-1, batchs, epochs, k, (10, 20, 10)))
        worker_list.append(Client(1, batchs, epochs, k, (10, 20)))
        communication_time = [0.0, 1000.0]
    
    client2server = [[] for i in range(client_num)]
    server2client = [[] for i in range(client_num)]
    
    
    max_time_per_batch = min((max_local_train+1) * (worker_list[i].training_time[0] + worker_list[i].training_time[1]) + 2 * communication_time[i] for i in range(client_num))
    floor_time_per_client = [(floor+1) * (worker_list[i].training_time[0] + worker_list[i].training_time[1]) + 2 * communication_time[i] for i in range(client_num)]
    max_time_per_batch = max(max(floor_time_per_client), max_time_per_batch)
    flexfl_per_client = [floor for i in range(client_num)]
    for i in range(client_num):
        temp = floor_time_per_client[i]
        while(temp < max_time_per_batch and flexfl_per_client[i] < max_local_train):
            temp += worker_list[i].training_time[0] + worker_list[i].training_time[1]
            if temp <= max_time_per_batch:
              flexfl_per_client[i] += 1
    
    Qs = []
    for i in range(1, client_num):
        Qs.append(flexfl_per_client[i])
    Qs.append(flexfl_per_client[0])
    
    print(f'comunication time is {communication_time}')
    return Qs
    
    

#def main(epochs=10, batchs=30, client_num=2, k=3, ceiling=4, timeout=10000000000):    
#def main(epochs=30, batchs=118, client_num=2, k=3, ceiling=4, timeout=10000000000):
#def main(epochs=30, batchs=59, client_num=2, k=3, ceiling=4, timeout=10000000000):
def main(epochs=30, batchs=391, client_num=4, k=3, ceiling=4, timeout=10000000000):
    Q, K = api4VFLAIR(epochs, batchs, client_num, k, ceiling)
    # for q in Q:
    #     print(q)
    # print(Q)
    
    # for k in K:
    #     print(k)
    # Q = api4FlexVFL(epochs, batchs, client_num, k, ceiling)
    print(Q)
    print(1)
    print(K)
    
    # a = 0
    # for k in K:
    #     print(a)
    #     for i in range(1, len(K[0])):
    #         if k[i] < k[i-1]:
    #             print("error")
    #     a += 1

if __name__ == '__main__':
    main()
