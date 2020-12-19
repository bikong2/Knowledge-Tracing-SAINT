import torch
import numpy as np
from torch import nn
import copy

"""
上三角
如果diagonal为空，输入矩阵保留主对角线与主对角线以上的元素；
如果diagonal为正数n，输入矩阵保留主对角线与主对角线以上除去n行的元素；
如果diagonal为负数-n，输入矩阵保留主对角线与主对角线以上与主对角线下方h行对角线的元素；
"""
def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
