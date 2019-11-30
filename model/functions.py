# -*- coding: utf-8 -*-
'''
Created on 2019-06-25, 10:50

@autorh: wangxb
'''

import torch
from torch.autograd import Function


class BinarizeF(Function):

    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# aliases
binarize = BinarizeF.apply
