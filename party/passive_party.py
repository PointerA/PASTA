import sys, os
sys.path.append(os.pardir)
from torch.utils.data import DataLoader
from party.party import Party
from dataset.party_dataset import PassiveDataset
from dataset.party_dataset import ActiveDataset
import queue

class PassiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.local_datas = queue.Queue()
        self.local_preds = queue.Queue()
        self.local_models = queue.Queue()
        self.local_gradients = queue.Queue()
        self.message_caches = {}
 
    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
            # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)

    def check_ready(self):
        if self.finish_batch+1 in self.message_caches:
            return True
        if self.local_gradients.empty():
            return False
        else:
            return True