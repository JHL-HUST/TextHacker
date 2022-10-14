import numpy as np
import torch
import torch.nn as nn


class BaseMethod(object):

    def __init__(self, model, use_gpu = False, device_id = [0]):
        super(BaseMethod,self).__init__()
        self.model = model
        self.use_gpu = use_gpu
        self.device_ids = device_id
        self.device = torch.device('cpu')
        if self.use_gpu:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])


    def attack(self, x):
        raise NotImplementedError


    def _model_into_cuda(self, model):
        if self.use_gpu:
            model = model.to(self.device)
            if len(self.device_ids) > 1:
                model = nn.DataParallel(model, device_ids = self.device_ids)
        return model
