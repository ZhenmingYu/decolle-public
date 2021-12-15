#!/bin/python
#-----------------------------------------------------------------------------
# File Name : lenet_decolle_model_fa.py
# Author: Emre Neftci
#
# Creation Date : Tue 04 Feb 2020 11:51:17 AM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from torch._C import iinfo
from .base_model import *
from .lenet_decolle_model import LenetDECOLLE
import math
from aihwkit.nn import AnalogLinear, AnalogConv2d

class LenetDECOLLEMemristor(LenetDECOLLE):
    '''
    Identical to LenetDECOLLE but with nn.layer replaced with memristor layers aihwkit.nn.layer
    '''
    def __init__(self, use_analog_layer = True, rpu_config = None, realistic_read_write = False, weight_scaling_omega = 0.0, *args, **kwargs):
        self.use_analog_layer = use_analog_layer
        self.rpu_config = rpu_config
        self.realistic_read_write = realistic_read_write
        self.weight_scaling_omega = weight_scaling_omega

        super(LenetDECOLLEMemristor, self).__init__(*args, **kwargs)

    def config_base_layer_Linear(self, in_features = None, out_features = None, bias=True):
        if in_features is None:
            raise ValueError('in_features is None')
        elif out_features is None:
            raise ValueError('out_features is None')
        elif self.use_analog_layer:
            base_layer_Linear = AnalogLinear(in_features, out_features, bias, self.rpu_config, self.realistic_read_write, self.weight_scaling_omega)
            # base_layer_Linear.weight=base_layer_Linear.weight.to("cuda")
        else:
            base_layer_Linear = nn.Linear(in_features, out_features, bias)
        return base_layer_Linear

    def config_base_layer_Conv2d(self, in_channels = None, out_channels = None, kernel_size = None, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        if in_channels is None:
            raise ValueError('in_channels is None')
        elif out_channels is None:
            raise ValueError('out_channels is None')
        elif kernel_size is None:
            raise ValueError('kernel_size is None')
        elif self.use_analog_layer:
            base_layer_Conv2d = AnalogConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, self.rpu_config, self.realistic_read_write, self.weight_scaling_omega)
            # base_layer_Conv2d.weight=base_layer_Conv2d.weight.to("cuda")
        else:
            base_layer_Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        return base_layer_Conv2d

    def build_conv_stack(self, Nhid, feature_height, feature_width, pool_size, kernel_size, stride, out_channels):
        output_shape = None
        padding = (np.array(kernel_size) - 1) // 2  
        for i in range(self.num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = self.config_base_layer_Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = self.lif_layer_type[i](base_layer,
                             alpha=self.alpha[i],
                             beta=self.beta[i],
                             alpharp=self.alpharp[i],
                             wrp=self.wrp[i],
                             deltat=self.deltat,
                             do_detach= True if self.method == 'rtrl' else False)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            if self.lc_ampl is not None:
                readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()
            self.readout_layers.append(readout)

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()


            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.dropout_layers.append(dropout_layer)
        return (Nhid[-1],feature_height, feature_width)

    def build_mlp_stack(self, Mhid, out_channels): 
        output_shape = None

        for i in range(self.num_mlp_layers):
            base_layer = self.config_base_layer_Linear(Mhid[i], Mhid[i+1])
            layer = self.lif_layer_type[i+self.num_conv_layers](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            if self.lc_ampl is not None:
                readout = nn.Linear(Mhid[i+1], out_channels)
                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, self.lc_ampl[i])
            else:
                readout = nn.Identity()

            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            output_shape = out_channels

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
        return (output_shape,)

    def build_output_layer(self, Mhid, out_channels):
        if self.with_output_layer:
            i=self.num_mlp_layers
            base_layer = self.config_base_layer_Linear(Mhid[i], out_channels)
            layer = self.lif_layer_type[-1](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         wrp=self.wrp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            readout = nn.Identity()
            if self.dropout[i] > 0.0:
                dropout_layer = nn.Dropout(self.dropout[i])
            else:
                dropout_layer = nn.Identity()
            
            output_shape = out_channels

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
        return (output_shape,)

    def get_input_layer_device(self):
        if hasattr(self.LIF_layers[0], 'get_device'):
            if self.LIF_layers[0].get_device() == torch.device('cpu'):
                print("Faking device type to 'cuda'")
                return torch.device('cuda')
            return self.LIF_layers[0].get_device() 
        else:
            return list(self.LIF_layers[0].parameters())[0].device

if __name__ == "__main__":
    #Test building network
    net = LenetDECOLLEMemristor(Nhid=[1,8],Mhid=[32,64],out_channels=10, input_shape=[1,28,28])
    d = torch.zeros([1,1,28,28])
    net(d)


