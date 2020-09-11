# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following lisence:

# Copyright 2020 Center for Biomedical Image Computing and Analytics, University of Pennsylvania
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# This is a 3-clause BSD license as defined in https://opensource.org/licenses/BSD-3-Clause

import torch.nn.functional as F
import torch.nn as nn
import torch

from fets.models.pytorch.brainmage.seg_modules import in_conv, DownsamplingModule, EncodingModule, InceptionModule, ResNetModule
from fets.models.pytorch.brainmage.seg_modules import UpsamplingModule, DecodingModule,IncDownsamplingModule,IncConv
from fets.models.pytorch.brainmage.seg_modules import out_conv, FCNUpsamplingModule, IncDropout,IncUpsamplingModule

from fets.models.pytorch.brainmage import BrainMaGeModel


"""
The smaller individual modules of these networks (the ones defined below), are taken from the seg_modules files as seen in the imports above.

This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The Downsampling, Encoding, Decoding modules
are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
"""

class PyTorch3DResUNet(BrainMaGeModel):
    def __init__(self, **kwargs):
        super(PyTorch3DResUNet, self).__init__(**kwargs)
        self.init_network(device=self.device)
        self.init_optimizer()
        

    def init_network(self, device, print_model=False, **kwargs):
        self.ins = in_conv(self.n_channels, self.base_filters, res=True)
        self.ds_0 = DownsamplingModule(self.base_filters, self.base_filters*2)
        self.en_1 = EncodingModule(self.base_filters*2, self.base_filters*2, res=True)
        self.ds_1 = DownsamplingModule(self.base_filters*2, self.base_filters*4)
        self.en_2 = EncodingModule(self.base_filters*4, self.base_filters*4, res=True)
        self.ds_2 = DownsamplingModule(self.base_filters*4, self.base_filters*8)
        self.en_3 = EncodingModule(self.base_filters*8, self.base_filters*8, res=True)
        self.ds_3 = DownsamplingModule(self.base_filters*8, self.base_filters*16)
        self.en_4 = EncodingModule(self.base_filters*16, self.base_filters*16, res=True)
        self.us_3 = UpsamplingModule(self.base_filters*16, self.base_filters*8)
        self.de_3 = DecodingModule(self.base_filters*16, self.base_filters*8, res=True)
        self.us_2 = UpsamplingModule(self.base_filters*8, self.base_filters*4)
        self.de_2 = DecodingModule(self.base_filters*8, self.base_filters*4, res=True)
        self.us_1 = UpsamplingModule(self.base_filters*4, self.base_filters*2)
        self.de_1 = DecodingModule(self.base_filters*4, self.base_filters*2, res=True)
        self.us_0 = UpsamplingModule(self.base_filters*2, self.base_filters)
        self.out = out_conv(self.base_filters*2, self.label_channels, self.binary_classification, res=True)

        if print_model:
            print(self)

        # send this to the device
        self.to(device)

    def forward(self, x):
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x