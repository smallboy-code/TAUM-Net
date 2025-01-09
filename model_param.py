
import os

import random
import numpy as np


import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim

from models.TransBraTS.TransBraTS_skipconnection import TransBraTS,Decoder_modual,mgmt_network,Grade_netwoek

from torchsummary import summary

from torchstat import stat



if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned").cuda()
    seg_model = Decoder_modual().cuda()
    mgmt_model = mgmt_network().cuda()


    # dict_model = {'en': model, 'seg': seg_model, 'mgmt': mgmt_model}
    # print(model)

    # summary(model, input_size=(4, 128, 128, 128), batch_size=1)
    # stat(model, input_size=(4, 128, 128, 128))

    print('ENCODER---------------------------------------------------')
    print(model)
    print('SEG---------------------------------------------------')
    print(seg_model)
    print('MGMT---------------------------------------------------')
    print(mgmt_model)






