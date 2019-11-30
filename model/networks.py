import torch
import torch.nn as nn
import functools
import numpy as np
import os
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
from model.attention import MultiHeadAttention
from model.base_model import clones, supported_acts, supported_loss, init_net, get_non_pad_mask, get_attn_pad_mask
from .functions import binarize

DEBUG_QUANTIZE = False
def set_debug_quentize(state = False):
    DEBUG_QUANTIZE = state

#############################################################################################################
################################################ Binary Linear ##############################################
#############################################################################################################
class BinaryLinear(nn.Linear):

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binarize(self.weight.org)

        output = nn.functional.linear(input, self.weight)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            output += self.bias.view(1, -1).expand_as(output)

        return output


#############################################################################################################
############################################# Deep FeedForward Layer ########################################
#############################################################################################################
class FCLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        act_func = nn.ReLU,
        batch_norm = False,
        dropout = 0.0,
        bias = True,
        binary = False,
        weight_init = None,
        bias_init = None):

        super(FCLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.binary = binary

        if self.binary:
            self.fc = BinaryLinear(input_size, output_size, bias = bias)
        else:
            self.fc = nn.Linear(input_size, output_size, bias = bias)

        if weight_init is not None:
            self.fc.weight.data.copy_(weight_init)

        if bias_init is not None:
            self.fc.bias.data.copy_(bias_init)

        #self.batch_norm = nn.LayerNorm(output_size) if batch_norm else None
        self.batch_norm = nn.BatchNorm1d(output_size) if batch_norm else None
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.bias = bias

        if DEBUG_QUANTIZE:
            self.fc_min        = 100000000
            self.fc_max        = -100000000
            self.fc_mean       = 0.0
            self.fc_std        = 0.0
            self.bn_min        = 100000000
            self.bn_max        = -100000000
            self.bn_mean       = 0.0
            self.bn_std        = 0.0
            self.act_min       = 100000000
            self.act_max       = -100000000
            self.act_mean      = 0.0
            self.act_std       = 0.0

    def forward(self, x):
        if len(x.size()) > 2:
            t, n, z = x.size(0), x.size(1), x.size(2)   # (num_block, num_frame, input_size)
            if not x.is_contiguous():
                x = x.contiguous()
            x = x.view(t * n, -1)                       # (num_block * num_frame, input_size)

        x = self.fc(x)

        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.fc_min  = min(x.min(), self.fc_min)
                self.fc_max  = max(x.max(), self.fc_max)
                self.fc_mean = (x.mean() +  self.fc_mean) / 2.0
                self.fc_std  = (x.std() +  self.fc_std) / 2.0

        if self.batch_norm is not None:
            x = self.batch_norm(x)

            if DEBUG_QUANTIZE:
                with torch.no_grad():
                    self.bn_min  = min(x.min(), self.bn_min)
                    self.bn_max  = max(x.max(), self.bn_max)
                    self.bn_mean = (x.mean() + self.bn_mean) / 2.0
                    self.bn_std  = (x.std() + self.bn_std) / 2.0

        if self.act_func is not None:
            x = self.act_func(x)

        if DEBUG_QUANTIZE:
            with torch.no_grad():
                self.act_min = min(x.min(), self.act_min)
                self.act_max = max(x.max(), self.act_max)
                self.act_mean = (x.mean() +  self.act_mean) / 2.0
                self.act_std = (x.std() +  self.act_std) / 2.0

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class DeepFFNet(nn.Module):
    "Define deep forward network to extract deep feature or solve classification or regression problem ."
    def __init__(
        self,
        layer_size,
        binary = False,
        hidden_act_type='relu',
        out_act_type = None,
        batch_norm = False,
        dropout = 0.0,
        bias = True):

        super(DeepFFNet, self).__init__()

        self.layer_size  = layer_size
        self.input_size  = layer_size[0]
        self.output_size = layer_size[-1]
        self.num_layer   = len(layer_size)
        self.batch_norm  = batch_norm
        self.bias        = bias
        self.dropout     = dropout
        self.binary      = binary

        if hidden_act_type is not None:
            self.hidden_act_type = hidden_act_type.lower()
            assert self.hidden_act_type in supported_acts, "act_type should be either relu, sigmoid, softmax or tanh"
            self.hidden_act = supported_acts[self.hidden_act_type]
        else:
            self.hidden_act = None

        if out_act_type is not None:
            self.out_act_type = out_act_type.lower()
            assert self.out_act_type in supported_acts, "act_type should be either relu, sigmoid, softmax or tanh"
            self.out_act = supported_acts[self.out_act_type]
        else:
            self.out_act = None

        layers = []

        for i in range(0, self.num_layer - 2):
            fc = FCLayer(
                    layer_size[i],
                    layer_size[i + 1],
                    act_func = self.hidden_act,
                    batch_norm = self.batch_norm,
                    dropout = self.dropout,
                    bias = self.bias,
                    binary = self.binary)
            layers.append(fc)

        fc = FCLayer(
                layer_size[-2],
                layer_size[-1],
                act_func = self.out_act,
                batch_norm = self.batch_norm,
                dropout = self.dropout,
                bias = self.bias,
                binary = self.binary)
        layers.append(fc)

        self.NNet = nn.Sequential(*layers)

    def forward(self, input):
        y = self.NNet(input)
        if len(input.size()) > 2:   # input: (num_block, num_frame, input_size)
            if not y.is_contiguous():
                y = y.contiguous()
            y = y.view(input.size()[0], input.size()[1], -1)
        return y


class Outputer(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(
        self,
        in_size,
        out_size,
        binary = False,
        out_act_type = None,
        batch_norm = False,
        bias = False):
        super(Outputer, self).__init__()

        self.binary = binary
        self.bias = bias
        self.batch_norm = batch_norm

        if out_act_type is not None:
            self.out_act_type = out_act_type.lower()
            assert self.out_act_type in supported_acts, "act_type should be either relu, sigmoid, softmax or tanh"
            self.out_act = supported_acts[self.out_act_type]
        else:
            self.out_act = None

        layers = []
        fc = FCLayer(
                in_size,
                out_size,
                act_func = self.out_act,
                batch_norm = self.batch_norm,
                dropout = 0.0,
                bias = self.bias,
                binary = self.binary)
        layers.append(fc)

        self.out_proj = nn.Sequential(*layers)

    def forward(self, x):
        y = self.out_proj(x)
        if len(x.size()) > 2:   # input: (num_block, num_frame, input_size)
            if not y.is_contiguous():
                y = y.contiguous()
            y = y.view(x.size()[0], x.size()[1], -1)
        return y


class DNNNet(nn.Module):
    "Define deep forward network to extract deep feature or solve classification or regression problem ."
    def __init__(
        self,
        layer_size,
        hidden_act_type='relu',
        out_act_type = None,
        batch_norm = False,
        dropout = 0.0):
        super(DNNNet, self).__init__()

        self.layer_size  = layer_size
        self.input_size  = layer_size[0]
        self.output_size = layer_size[-1]
        self.num_layer   = len(layer_size)
        self.batch_norm  = batch_norm
        self.dropout     = dropout

        self.DeepFF = DeepFFNet(
            layer_size = layer_size[0:-1],
            hidden_act_type = hidden_act_type,
            out_act_type = hidden_act_type,
            batch_norm = batch_norm,
            dropout = dropout)

        self.Output = Outputer(
            in_size = layer_size[-2],
            out_size = layer_size[-1],
            out_act_type = out_act_type,
            bias = False)

    def forward(self, x):
        last_h = self.DeepFF(x)
        y = self.Output(last_h)
        return y, last_h


#############################################################################################################
############################################# Transformer Layer #############################################
#############################################################################################################
class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class TransformerLayer(nn.Module):
    "DeepTransformer is made up of self-attn and feed forward (defined below)"
    def __init__(
        self,
        in_size,
        d_model,
        query_key_size,
        value_size,
        num_head = 1,
        layer_norm = False,
        residual_op = False,
        dropout = 0.0):
        super(TransformerLayer, self).__init__()

        self.residual_op = residual_op
        if residual_op:
            assert d_model == in_size, "d_model = %d and in_size = %d MUST be same for the self-attention with residual_op" % (in_size, d_model)

        self.self_attn = MultiHeadAttention(
            n_head = num_head,
            in_d_q = in_size,
            in_d_k = in_size,
            in_d_v = in_size,
            out_d_k = query_key_size,
            out_d_v = value_size,
            out_d = d_model,
            dropout = 0.1)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_norm = None

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, non_pad_mask = None, slf_attn_mask = None):
        "Follow Figure 1 (left) for connections."

        residual = x

        x, attn = self.self_attn( x, x, x, mask = slf_attn_mask)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.residual_op:
            x = x + residual

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if non_pad_mask is not None:
            x *= non_pad_mask

        return x, attn


class TransformerFFLayer(nn.Module):
    "DeepTransformer is made up of self-attn and feed forward (defined below)"
    def __init__(
        self,
        in_size,
        d_model,
        d_ff_inner,
        query_key_size,
        value_size,
        num_head = 1,
        layer_norm = False,
        residual_op = False,
        dropout = 0.0):
        super(TransformerFFLayer, self).__init__()

        self.residual_op = residual_op
        if residual_op:
            assert d_model == in_size, "d_model = %d and in_size = %d MUST be same for the self-attention with residual_op" % (in_size, d_model)

        self.self_attn = MultiHeadAttention(
            n_head = num_head,
            in_d_q = in_size,
            in_d_k = in_size,
            in_d_v = in_size,
            out_d_k = query_key_size,
            out_d_v = value_size,
            out_d = d_model,
            dropout = 0.1)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff_inner, dropout = dropout)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_norm = None

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, non_pad_mask = None, slf_attn_mask = None):
        "Follow Figure 1 (left) for connections."

        residual = x

        x, attn = self.self_attn( x, x, x, mask = slf_attn_mask)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.residual_op:
            x = x + residual

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if non_pad_mask is not None:
            x *= non_pad_mask
        x = self.pos_ffn(x)
        if non_pad_mask is not None:
            x *= non_pad_mask

        return x, attn


#############################################################################################################
############################################# Fbank Layer ###################################################
#############################################################################################################
def to_cuda(m, x):
    assert isinstance(m, torch.nn.Module)
    device_id = torch.cuda.device_of(next(m.parameters()).data).idx
    if device_id == -1:
        return x
    return x.cuda(device_id)


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595. * np.log10(1+hz/700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700.0*(10.0**(mel/2595.0)-1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1) * mel2hz(melpoints)/ float(samplerate))

    fbank = np.zeros([nfilt, nfft//2+1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    fbank = fbank[:, 0:nfft//2]
    return fbank


class FbankModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(FbankModel, self).__init__()

        filterbanks = get_filterbanks(
            nfilt = out_size,
            nfft = 512,
            samplerate = 16000,
            lowfreq = 80,
            highfreq = 8000) # out_size, in_size
        filterbanks[1, 4] = 1.0

        self.fbank = nn.Linear(in_size, out_size, bias=False)
        self.fbank.weight.data.copy_(torch.from_numpy(filterbanks))

    def forward(self, audio_spect, fbank_cmvn=None):
        '''FbankModel forward
        :param xs:
        :return:
        '''
        # [num_block, 1, block_length, block_width]

        num_block, num_channel, num_frame = audio_spect.size(0), audio_spect.size(1), audio_spect.size(2)
        if not audio_spect.is_contiguous():
            audio_spect = audio_spect.contiguous()

        #audio_spect = 1.0 / 512.0 * (audio_spect ** 2)
        audio_spect = 10000.0 * (audio_spect ** 2)
        audio_spect = audio_spect.view(num_block * num_channel * num_frame, -1)
        out = self.fbank(audio_spect)
        out = torch.clamp(out, min=1e-13)
        out = torch.log(out) / 10.0

        if fbank_cmvn is not None:
            fbank_cmvn = to_cuda(self, fbank_cmvn)
            out = (out + fbank_cmvn[0, :]) * fbank_cmvn[1, :]

        out = out.view(num_block, num_channel, num_frame, -1)
        return out


class FbankBaseModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(FbankBaseModel, self).__init__()

        filterbanks = get_filterbanks(
            nfilt = out_size,
            nfft = 512,
            samplerate = 16000,
            lowfreq = 80,
            highfreq = 8000) # out_size, in_size
        filterbanks[1, 4] = 1.0
        self.fbank = nn.Linear(in_size, out_size, bias=False)
        self.fbank.weight.data.copy_(torch.from_numpy(filterbanks))

    def forward(self, audio_spect, fbank_cmvn=None):
        '''FbankModel forward
        :param xs:
        :return:
        '''
        audio_spect = 10000.0 * (audio_spect ** 2)
        out = self.fbank(audio_spect)
        out = torch.clamp(out, min=1e-13)
        out = torch.log(out) / 10.0

        if fbank_cmvn is not None:
            fbank_cmvn = to_cuda(self, fbank_cmvn)
            out = (out + fbank_cmvn[0, :]) * fbank_cmvn[1, :]

        return out
