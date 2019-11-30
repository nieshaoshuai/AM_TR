import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import math, copy, time
import torch.nn as nn
from torch.nn import init
import numpy as np
from model.scheduler import get_scheduler
import codecs

supported_rnns = {'lstm': nn.LSTM, 'rnn': nn.RNN, 'gru': nn.GRU}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())
supported_acts = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(), 'tanh': nn.Tanh(),'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU(), 'softplus': nn.Softplus()}
supported_acts_inv = dict((v, k) for k, v in supported_acts.items())
supported_loss = {'mseloss': nn.MSELoss(), 'kldivloss': nn.KLDivLoss(), 'smoothl1loss': nn.SmoothL1Loss()}
supported_loss_inv = dict((v, k) for k, v in supported_loss.items())


def ChooseQuantizationQParams(vmax, qmax):
    vmax = np.abs(vmax)
    Q = 0
    if vmax < qmax:
        while vmax * 2 <= qmax:
            Q = Q + 1.0
            vmax = vmax * 2.0
    else:
        while  vmax >= qmax:
            Q = Q - 1
            vmax = vmax * 0.5
    return Q

def QQuantize(qparams, fdata, qBit):
    if len(fdata.shape) < 2:
        fdata = fdata[np.newaxis, :]
    row, col = fdata.shape
    if qBit == 8:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int8)
    elif qBit == 16:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int16)
    elif qBit == 32:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int32)
    elif qBit == 64:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int64)
    else:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int8)
    
    for i in range(row):
        for j in range(col):
            real_val = fdata[i, j]
            transformed_val = real_val * 2**qparams
            clamped_val = max( -2**(qBit-1),  min( 2**(qBit-1) - 1, transformed_val ))

            if qBit == 8:
                fixed_data[i, j]  = np.int8(np.round(clamped_val))
            elif qBit == 16:
                fixed_data[i, j]  = np.int16(np.round(clamped_val))
            elif qBit == 32:
                fixed_data[i, j]  = np.int32(np.round(clamped_val))
            elif qBit == 64:
                fixed_data[i, j]  = np.int64(np.round(clamped_val))
            else:
                fixed_data[i, j]  = np.int8(np.round(clamped_val))
    return fixed_data

def aQQuantize(qparams, fdata, bits):

    transformed_val = fdata * 2 ** qparams
    clamped_val = max( -2**(bits-1),  min( 2**(bits-1) - 1, transformed_val ))
    if bits == 8:
        fixed_data  = np.int8(np.round(clamped_val))
    elif bits == 16:
        fixed_data  = np.int16(np.round(clamped_val))
    elif bits == 32:
        fixed_data  = np.int32(np.round(clamped_val))
    elif bits == 64:
        fixed_data  = np.int64(np.round(clamped_val))
    else:
        fixed_data  = np.int8(np.round(clamped_val))
    return fixed_data

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths) # num_block, num_frame
    # N x Ti, lt(1) like not operation
    #pad_mask = non_pad_mask.squeeze(-1).lt(1)
    pad_mask = non_pad_mask.squeeze(-1).gt(0) #num_block, num_frame
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1) # num_block, 1, num_frame
    return attn_mask


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, device = None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    
    '''if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        net.to(device)
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    '''
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.continue_from_name = opt.continue_from_name
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = opt.model_dir      # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        #self.setup(opt)

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        #if self.continue_from_name:
        #    self.load_networks(self.continue_from_name)

        self.print_networks(opt.verbose)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()
            
        lr = self.optimizers[0].param_groups[0]['lr']
        #print('learning rate = %.7f' % lr)
        
    def set_lr_factor(self, lr_factor):
        """Update learning rates for all the networks; called at the end of every epoch"""
        if self.opt.lr_policy == 'warmup':
            for scheduler in self.schedulers:
                scheduler.set_lr_factor(lr_factor)
        
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def train(self):
        """Make models train mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, suffix_name):
        """Save all the networks to the disk.
        Parameters:
            suffix_name (int) -- current epoch; used in the file name '%s_net_%s.pth' % (suffix_name, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pth' % (name, suffix_name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    #torch.save(net.modules.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    
    def load_networks(self, suffix_names):
        suffix_names = suffix_names.split('-')
        num_model = len(suffix_names)
        if len(suffix_names) == 1:
            suffix_names = suffix_names[0]
        
        print(suffix_names)
        print("num_model = %d" % (num_model))
        
        if isinstance(suffix_names, list):
            nets_dict = {}
            for suffix_name in suffix_names:
                for name in self.model_names:
                    if isinstance(name, str):
                        load_filename = '%s_%s.pth' % (name, suffix_name)
                        load_path = os.path.join(self.save_dir, load_filename)
                        print('average_model[%s]: loading the model from %s' % (name, load_path))
                        
                        cur_state_dict = torch.load(load_path, map_location=str(self.device))
                        if hasattr(cur_state_dict, '_metadata'):
                            del cur_state_dict._metadata
                        
                        if name in nets_dict.keys():
                            state_dict = nets_dict[name]
                            for key in state_dict.keys():
                                if key in cur_state_dict.keys():
                                    print("--> + %s" % (key))
                                    state_dict[key] += cur_state_dict[key]
                            nets_dict[name] = state_dict
                        else:
                            nets_dict[name] = cur_state_dict
                            
            for name in nets_dict.keys():
                state_dict = nets_dict[name]
                for key in state_dict.keys():
                    state_dict[key] /= num_model
                
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.modules
                    
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                
                net.load_state_dict(state_dict, strict=False)
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    load_filename = '%s_%s.pth' % (name, suffix_names)
                    load_path = os.path.join(self.save_dir, load_filename)
                    net = getattr(self, name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.modules
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
    
                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict, strict=False)
    
    '''
    def load_networks(self, suffix_name):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (name, suffix_name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.modules
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict, strict=False)
    '''
    
    def init_model(self, model_path, continue_from_name = 'best'):
        if continue_from_name is None:
            print("ERROR: continue_from_model is None")
            return False
        
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (name, continue_from_name)
                load_path = os.path.join(model_path, load_filename)
                
                if os.path.exists(load_path):
                    print("initlizing %s with %s" % (name, load_path))
                    
                    state_dict = torch.load(load_path, map_location = self.device)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    net = getattr(self, name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.modules
                    
                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                        
                    net.load_state_dict(state_dict, strict=False)
                    
    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def write_to_kaldi(self, feat_size, cmvn_file, suffix_name):
        
        cmvn = None
        if os.path.exists(cmvn_file):
            cmvn = np.load(cmvn_file)
            cmvn_size = np.shape(cmvn)[1]
        
        save_filename = '%s.kaldi' % (suffix_name)
        out_file_name = os.path.join(self.save_dir, save_filename)
        f = codecs.open(out_file_name, 'w', 'utf-8')

        if cmvn is not None:
            f.write('<normalization> {0} {1}\n'.format(feat_size, feat_size))
            f.write('\t<addshift> {0} {1}\n'.format(feat_size, feat_size))
            f.write('\t[ ')
            for i in range(feat_size):
                f.write('%.9f ' % cmvn[0, i])
            f.write(']\n')
            f.write('\t<rescale> {0} {1}\n'.format(feat_size, feat_size))
            f.write('\t[ ')
            for i in range(feat_size):
                f.write('%.9f ' % cmvn[1, i])
            f.write(']\n')
        f.write('<input> {0} {1}\n'.format(cmvn_size, feat_size))
        
        for name in self.model_names:
            if isinstance(name, str):
                if name == "out_prj":
                    net = getattr(self, name)
                    net_dict = net.cpu().state_dict()
                    for k, v in net_dict.items():
                        if v.is_cuda:
                            v = v.cpu()

                        if len(v.shape) >= 2:
                            in_size, out_size = v.shape
                            f.write('{0} <AffineTransform> {1} {2}\n'.format(k, in_size, out_size))
                            f.write('<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0  [\n')
                            for i in range(in_size - 1):
                                f.write('  ')
                                for j in range(out_size):
                                    f.write('%.9f ' % v[i, j])
                                f.write('\n')
                            f.write('  ')
                            for j in range(out_size):
                                f.write('%.9f ' % v[in_size - 1, j])
                            f.write(']\n')
                        else:
                            in_size = v.shape[0]
                            f.write('{0} <bias> {1} {2} [ '.format(k, in_size, in_size))
                            for i in range(in_size):
                                f.write('%.9f ' % v[i])
                            f.write(']\n')

                        f.write('<softmax> {0} {1}\n'.format(in_size, in_size))
                    f.write('<output> {0} {1}\n'.format(in_size, in_size))
                else:
                    configure = getattr(self, "%s_args" % name)

                    net = getattr(self, name)
                    net_dict = net.cpu().state_dict()

                    for k, v in net_dict.items():
                        if v.is_cuda:
                            v = v.cpu()
                            
                        if len(v.shape) >= 2:
                            in_size, out_size = v.shape
                            f.write('{0} <AffineTransform> {1} {2}\n'.format(k, in_size, out_size))
                            f.write('<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0  [\n')
                            for i in range(in_size - 1):
                                f.write('  ')
                                for j in range(out_size):
                                    f.write('%.9f ' % v[i, j])
                                f.write('\n')
                            f.write('  ')
                            for j in range(out_size):
                                f.write('%.9f ' % v[in_size - 1, j])
                            f.write(']\n')
                        else:
                            in_size = v.shape[0]
                            f.write('{0} <bias> {1} {2} [ '.format(k, in_size, in_size))
                            for i in range(in_size):
                                f.write('%.9f ' % v[i])
                            f.write(']\n')

                        if configure.hidden_act_type is not None:
                            f.write('<{0}> {1} {2}\n'.format(configure.hidden_act_type, in_size, in_size))
        f.close()

    def write_to_ccode(self, cmvn_file, out_file_name, basic_file_name, feat_size = 40, input_size = 440):
        
        ## integrete the cmvn into the weight and bias of 1st layer ##
        cmvn = None
        if os.path.exists(cmvn_file):
            cmvn = np.load(cmvn_file)
            
        feat_size = cmvn.shape[1]
        context_frame = int(input_size / feat_size)
        if cmvn is not None:
            addshift = cmvn[0, :]
            rescale  = cmvn[1, :]
            
            feat_delta = 1.0 / rescale
            feat_u     = -addshift

            delta = np.tile(feat_delta, [1, context_frame])
            u     = np.tile(feat_u, [1, context_frame])
        else:
            delta = None
            u     = None

        ## quantize the weight and bias ##
        qmax = 127
        quantized_net = []
        if 'deepff' in self.model_names:
            net = getattr(self, 'deepff')
            net_dict = net.cpu().state_dict()

            if 'NNet.0.fc.weight' in net_dict and 'NNet.0.fc.bias' in net_dict:
                weight = net_dict['NNet.0.fc.weight'].cpu().numpy() # 400 * 440
                bias   = net_dict['NNet.0.fc.bias'].cpu().numpy()   # 400 * 1
                bias   = bias[:, np.newaxis]

                weight = weight / delta
                bias   = bias - weight.dot(u.T)                     # [400, 440] * [440, 1]
                bias   = bias.squeeze()

                net_dict['NNet.0.fc.weight'] = torch.from_numpy(weight)
                net_dict['NNet.0.fc.bias']   = torch.from_numpy(bias)

            net_layers = {}
            for k in net_dict.keys():
                ilayer = int(k.split('.')[1])
                if ilayer in net_layers:
                    layer = net_layers[ilayer]
                    layer.append(k)
                    net_layers[ilayer] = layer
                else:
                    net_layers[ilayer] = [k]

            for ilayer, layers in net_layers.items():
                weight = None
                bias = None
                for k in layers:
                    if 'weight' in k:
                        weight = net_dict[k].cpu().numpy()
                    if 'bias' in k:
                        bias = net_dict[k].cpu().numpy()
                
                if weight is not None:
                    if bias is not None:
                        qparams = np.zeros(shape = weight.shape[0], dtype = np.int)
                        qweight = np.zeros(shape = weight.shape, dtype=np.int8)
                        qbias   = np.zeros(shape = bias.shape, dtype=np.int32)
                        for r in range(weight.shape[0]):
                            w_abs_max = max(np.max(np.abs(weight[r, :])), 0.001)
                            params    = ChooseQuantizationQParams(w_abs_max, qmax)
                            qweight[r, :] = QQuantize(params, weight[r, :], 8)
                            qbias[r] = aQQuantize(params, bias[r], 32)
                            qparams[r] = params
                    else:
                        qparams = np.zeros(shape = weight.shape[0], dtype = np.float)
                        qweight = np.zeros(shape = weight.shape, dtype = np.int8)
                        qbias   = None
                        for r in range(weight.shape[0]):
                            w_abs_max = max(np.max(np.abs(weight[r, :])), 0.001)
                            params    = ChooseQuantizationQParams(w_abs_max, qmax)
                            qweight[r, :] = QQuantize(params, weight[r, :], 8)
                            qparams[r] = params
                else: 
                    qparams = None
                    qweight = None
                    qbias   = None
                
                quantized_net.append((qweight, qbias, qparams))
        
        if 'out_prj' in self.model_names:
            net = getattr(self, 'out_prj')
            net_dict = net.cpu().state_dict()

            net_layers = {}
            for k in net_dict.keys():
                ilayer = int(k.split('.')[1])
                if ilayer in net_layers:
                    layer = net_layers[ilayer]
                    layer.append(k)
                    net_layers[ilayer] = layer
                else:
                    net_layers[ilayer] = [k]

            for ilayer, layers in net_layers.items():
                weight = None
                bias = None
                for k in layers:
                    if 'weight' in k:
                        weight = net_dict[k].cpu().numpy()
                    if 'bias' in k:
                        bias = net_dict[k].cpu().numpy()
                
                if weight is not None:
                    if bias is not None:
                        qparams = np.zeros(shape = weight.shape[0], dtype = np.int)
                        qweight = np.zeros(shape = weight.shape, dtype=np.int8)
                        qbias   = np.zeros(shape = bias.shape, dtype=np.int32)
                        for r in range(weight.shape[0]):
                            w_abs_max = max(np.max(np.abs(weight[r, :])), 0.001)
                            params    = ChooseQuantizationQParams(w_abs_max, qmax)
                            qweight[r, :] = QQuantize(params, weight[r, :], 8)
                            qbias[r] = aQQuantize(params, bias[r], 32)
                            qparams[r] = params
                    else:
                        qparams = np.zeros(shape = weight.shape[0], dtype = np.float)
                        qweight = np.zeros(shape = weight.shape, dtype=np.int8)
                        qbias   = None
                        for r in range(weight.shape[0]):
                            w_abs_max = max(np.max(np.abs(weight[r, :])), 0.001)
                            params    = ChooseQuantizationQParams(w_abs_max, qmax)
                            qweight[r, :] = QQuantize(params, weight[r, :], 8)
                            qparams[r] = params
                else: 
                    qparams = None
                    qweight = None
                    qbias   = None
                
                quantized_net.append((qweight, qbias, qparams))
        
        ## write header file ##
        f = codecs.open(out_file_name, 'w', 'utf-8')

        f.write('#include \"skv_layers.h\"\n')
        f.write('#include \"../basic/os_support.h\"\n')
        f.write('#include \"../math/skv_math_core.h\"\n')
        f.write('#include \"../math/skv_fastmath.h\"\n')
        f.write('#include <stdio.h>\n')
        f.write('#include <assert.h>\n')
        f.write('#include <stdbool.h>\n\n')

        ## write the weight and bias ##
        for ilayer in range(len(quantized_net)):
            qweight, qbias, qparams = quantized_net[ilayer]
            f.write('static const skv_weight awakeup_weight_%d[%d] = {' % (ilayer + 1, qweight.shape[0] * qweight.shape[1]))
            num_w = 0
            num = qweight.shape[0] * qweight.shape[1]
            for i in range(qweight.shape[0]):
                for j in range(qweight.shape[1]):
                    if num_w % 8 == 0:
                        if num_w == num - 1:
                            f.write('\n\t%d ' % qweight[i, j])
                        else:
                            f.write('\n\t%d, ' % qweight[i, j])
                        num_w = num_w + 1
                    else:
                        if num_w == num - 1:
                            f.write('%d ' % qweight[i, j])
                        else:
                            f.write('%d, ' % qweight[i, j])
                        num_w = num_w + 1
            f.write('\n\t};\n')

            f.write('static const skv_bias awakeup_bias_%d[%d] = {' % (ilayer + 1, qbias.shape[0]))
            num_b = 0
            num = qbias.shape[0]
            for i in range(qbias.shape[0]):
                if num_b % 8 == 0:
                    if num_b == num - 1:
                        f.write('\n\t%d ' % (qbias[i]))
                    else:
                        f.write('\n\t%d, ' % (qbias[i]))
                    num_b = num_b + 1
                else:
                    if num_b == num - 1:
                        f.write('%d ' % (qbias[i]))
                    else:
                        f.write('%d, ' % (qbias[i]))
                    num_b = num_b + 1
            f.write('\n\t};\n\n')

        ## write the layer out min_max value ##
        deepff = getattr(self, 'deepff')
        num_layer = len(deepff.NNet)
        f.write('static const float layer_out_min[%d] = { ' % (num_layer + 3))
        for i in range(num_layer + 2):
            f.write('0.0f, ')
        f.write('0.0f };\n')

        f.write('static const float layer_out_max[%d] = { 100.0f, ' % (num_layer + 3))
        for layer in range(num_layer):
            act_mean = deepff.NNet[layer].act_mean
            act_std  = deepff.NNet[layer].act_std
            f.write('%.3f, ' % (act_mean + 3.0 * act_std))
        out_prj = getattr(self, 'out_prj')
        act_mean = out_prj.out_proj[0].act_mean
        act_std  = out_prj.out_proj[0].act_std
        f.write('%.3f }; \n' % (act_mean + 3.0 * act_std))

        ## write the basic code from the basic code file ##
        with open(basic_file_name) as freader:
            for line in freader.readlines():
                f.write('%s' % (line))
        
        num_layer = len(deepff.NNet) + len(out_prj.out_proj) + 1 + 1 + 1
        f.write('\nEXPORT SKVLayerState * skv_asr_layers_init()\n')
        f.write('{\n')

        f.write('\tSKVLayerState * st = NULL;\n')
        f.write('\tst = (SKVLayerState *)speex_alloc(sizeof(SKVLayerState));\n')
        f.write('\tst->num_layers = %d;\n' % (num_layer))
        f.write('\tst->layers = (void **)speex_alloc(st->num_layers * sizeof(void *));\n\n')
        
        f.write('\tBasicLayer        * basic_layer           = NULL;\n')
        f.write('\tActiveLayer       * active_layer          = NULL;\n')
        f.write('\tAffineLayer       * affine_layer          = NULL;\n')
        f.write('\tAffineActiveLayer * affine_active_layer   = NULL;\n')
        f.write('\tint layer = 0;\n')

        f.write('\n\n')
        f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(BasicLayer));\n')
        f.write('\tbasic_layer = (BasicLayer *)st->layers[layer];\n')
        f.write('\tbasic_layer->layer_type = INPUTLayer;\n')
        f.write('\tbasic_layer->pre_ptr  = layer - 1;\n')
        f.write('\tbasic_layer->next_ptr = layer + 1;\n')
        f.write('\tbasic_layer->in_size     = %d;\n' % (feat_size))
        f.write('\tbasic_layer->out_size    = %d;\n' % (input_size))
        f.write('\tbasic_layer->isQuantized = false;\n')
        f.write('\tbasic_layer->out_qParams = 0;\n')
        f.write('\tlayer++;\n')

        hidden_act = deepff.hidden_act_type
        for i in range(len(deepff.NNet)):
            num_affine = i + 1
            qweight, qbias, qparams = quantized_net[i]
            outsize, insize = qweight.shape
            if hidden_act == 'relu':
                f.write('\n\n')
                f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(AffineActiveLayer));\n')
                f.write('\taffine_active_layer = (AffineActiveLayer *)st->layers[layer];\n')
                f.write('\taffine_active_layer->layer_type = AFFINEACTIVELayer;\n')
                f.write('\taffine_active_layer->pre_ptr = layer - 1;\n')
                f.write('\taffine_active_layer->next_ptr = layer + 1;\n')
                f.write('\taffine_active_layer->in_size    = %d;\n' % (insize))
                f.write('\taffine_active_layer->out_size   = %d;\n' % (outsize))
                f.write('\taffine_active_layer->isQuantized= true;\n')
                f.write('\taffine_active_layer->out_qParams= 0;\n')
                f.write('\taffine_active_layer->active_type= ReLU;\n')
                f.write('\tlayer++;\n')
                f.write('\taffine_active_layer->layer_w = awakeup_weight_%d;\n' % (num_affine))
                f.write('\taffine_active_layer->layer_b = awakeup_bias_%d;\n' % (num_affine))
                f.write('\taffine_active_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (outsize))
                for r in range(outsize):
                    f.write('\taffine_active_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))
            else:
                f.write('\n\n')
                f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(AffineLayer));\n')
                f.write('\taffine_layer = (AffineLayer *)st->layers[layer];\n')
                f.write('\taffine_layer->layer_type = AFFINELayer;\n')
                f.write('\taffine_layer->pre_ptr = layer - 1;\n')
                f.write('\taffine_layer->next_ptr = layer + 1;\n')
                f.write('\taffine_layer->in_size    = %d;\n' % (insize))
                f.write('\taffine_layer->out_size   = %d;\n' % (outsize))
                f.write('\taffine_layer->isQuantized= true;\n')
                f.write('\taffine_layer->out_qParams= 0;\n')
                f.write('\tlayer++;\n')
                f.write('\taffine_layer->layer_w = awakeup_weight_%d;\n' % (num_affine))
                f.write('\taffine_layer->layer_b = awakeup_bias_%d;\n' % (num_affine))
                f.write('\taffine_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (outsize))
                for r in range(outsize):
                    f.write('\taffine_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))

            if hidden_act == 'sigmoid':
                act_func = 'Sigmoid'
            elif hidden_act == 'tanh':
                act_func = 'Tanh'
            elif hidden_act == 'softmax':
                act_func = 'Softmax'
            else:
                act_func = None
            
            if act_func is not None:
                f.write('\n\n')
                f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(ActiveLayer));\n')
                f.write('\tactive_layer = (ActiveLayer *)st->layers[layer];\n')
                f.write('\tactive_layer->layer_type = ACTIVELayer;\n')
                f.write('\tactive_layer->pre_ptr = layer - 1;\n')
                f.write('\tactive_layer->next_ptr = layer + 1;\n')
                f.write('\tactive_layer->in_size    = %d;\n' % (outsize))
                f.write('\tactive_layer->out_size   = %d;\n' % (outsize))
                f.write('\taffine_layer->isQuantized= false;\n')
                f.write('\tactive_layer->out_qParams= 0;\n')
                f.write('\tactive_layer->active_type = %s;\n' % (act_func))
                f.write('\tlayer++;\n')
        
        num_affine = num_affine + 1
        qweight, qbias, qparams = quantized_net[-1]
        outsize, insize = qweight.shape

        f.write('\n\n')
        f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(AffineLayer));\n')
        f.write('\taffine_layer = (AffineLayer *)st->layers[layer];\n')
        f.write('\taffine_layer->layer_type = AFFINELayer;\n')
        f.write('\taffine_layer->pre_ptr = layer - 1;\n')
        f.write('\taffine_layer->next_ptr = layer + 1;\n')
        f.write('\taffine_layer->in_size    = %d;\n' % (insize))
        f.write('\taffine_layer->out_size   = %d;\n' % (outsize))
        f.write('\taffine_layer->isQuantized= true;\n')
        f.write('\taffine_layer->out_qParams= 0;\n')
        f.write('\tlayer++;\n')
        f.write('\taffine_layer->layer_w = awakeup_weight_%d;\n' % (num_affine))
        f.write('\taffine_layer->layer_b = awakeup_bias_%d;\n' % (num_affine))
        f.write('\taffine_layer->layer_qParams = (skv_int16_t *)speex_alloc( %d * sizeof(skv_int16_t));\n' % (outsize))
        for r in range(outsize):
            f.write('\taffine_layer->layer_qParams[%d] = %d;\n' % (r, qparams[r]))

        f.write('\n\n')
        f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(ActiveLayer));\n')
        f.write('\tactive_layer = (ActiveLayer *)st->layers[layer];\n')
        f.write('\tactive_layer->layer_type = ACTIVELayer;\n')
        f.write('\tactive_layer->pre_ptr = layer - 1;\n')
        f.write('\tactive_layer->next_ptr = layer + 1;\n')
        f.write('\tactive_layer->in_size    = %d;\n' % (outsize))
        f.write('\tactive_layer->out_size   = %d;\n' % (outsize))
        f.write('\taffine_layer->isQuantized= false;\n')
        f.write('\tactive_layer->out_qParams= 0;\n')
        f.write('\tactive_layer->active_type = Softmax;\n')
        f.write('\tlayer++;\n')

        f.write('\n\n')
        f.write('\tst->layers[layer] = (void *)speex_alloc(sizeof(BasicLayer));\n')
        f.write('\tbasic_layer = (BasicLayer *)st->layers[layer];\n')
        f.write('\tbasic_layer->layer_type = OUTPUTLayer;\n')
        f.write('\tbasic_layer->pre_ptr = layer - 1;\n')
        f.write('\tbasic_layer->next_ptr = layer + 1;\n')
        f.write('\tbasic_layer->in_size    = %d;\n' % (outsize))
        f.write('\tbasic_layer->out_size   = %d;\n' % (outsize))
        f.write('\tbasic_layer->isQuantized= false;\n')
        f.write('\tbasic_layer->out_qParams= 0;\n')
        f.write('\tlayer++;\n')
        
        f.write('\n\n')
        f.write('\tif (preComputeLayerParam(st) == false)\n')
        f.write('\t{\n')
        f.write('\t\tst = skv_layers_destroy(st);\n')
        f.write('\t\tst = NULL;\n')
        f.write('\t}\n')
        f.write('\treturn st;\n')
        f.write('}\n')

        f.close()