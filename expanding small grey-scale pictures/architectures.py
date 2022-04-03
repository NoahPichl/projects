# -*- coding: utf-8 -*-
"""example_project/architectures.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.02.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Architectures file of example project.
"""

import torch


class CNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 2, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(CNN, self).__init__()
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size/2)))
            cnn.append(torch.nn.SELU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size/2))
    
    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred


class CNN_2(torch.nn.Module):
    def __init__(self, n_in_channels: int = 2, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(CNN_2, self).__init__()

        self.conv_1 = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=16, kernel_size=5, bias=True,
                                      padding=2)
        self.conv_2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, bias=True,
                                        padding=2)
        self.conv_3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, bias=True,
                                      padding=2)
        self.output_layer = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, bias=True,
                                            padding=2)

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        pred = self.output_layer(x)
        return pred