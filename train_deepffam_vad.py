from __future__ import print_function
import argparse
import os
import random
import shutil
import psutil
import time 
import gc
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.autograd import Variable

from trainer import AMTrainer
from model.model import DeepFFAM, DeepFFTransformerAM, DeepFFTransformerDeepFFAM
from data.data_new_new_new import BlockDataset, BlockDataLoader, BlockBatchGenerator
from utils import utils 
import numpy as np

basic_parser = argparse.ArgumentParser(description='Awakeup AM Training')

basic_parser.add_argument('--work_name', default='Noisy_VAD', type=str, help='the description to this work')
basic_parser.add_argument('--works_dir', metavar='DIR', help='the path to main dir', default="/home/snie/works/train_am/am_train_new/egs/vad/")

# data configures
basic_parser.add_argument('--speech_dir', metavar='DIR', help='path to clean speech', default='/home/snie/works/train_am/am_train_new/egs/vad/data/speech')
basic_parser.add_argument('--noisy_dir', metavar='DIR', help='path to noisy speech', default='/home/snie/works/train_am/am_train_new/egs/vad/data/noisy')
basic_parser.add_argument('--noise_dir', metavar='DIR', help='path to noise', default='/home/snie/works/train_am/am_train_new/egs/vad/data/noise')

basic_parser.add_argument('--specially_speech_dir', metavar='DIR', help='path to clean speech', default=None)
basic_parser.add_argument('--specially_noisy_dir', metavar='DIR', help='path to noisy speech', default=None)
basic_parser.add_argument('--specially_noise_dir', metavar='DIR', help='path to noise', default=None)

basic_parser.add_argument('--num_workers', type=int, default=2, help='the num_workers')
basic_parser.add_argument('--num_utt_per_loading', type=int, default=30, help='the number of utterances for each loading')

basic_parser.add_argument('--pos_batch_size', type=int, default=500, help='the batch size for once training')
basic_parser.add_argument('--neg_batch_size', type=int, default=500, help='the batch size for once training')

basic_parser.add_argument('--block_length', default=1, type=int, help='the size of block')
basic_parser.add_argument('--block_shift', default=1, type=int, help='the shift of block')

basic_parser.add_argument('--delta_order', default=0, type=int, help='Delta order for training')
basic_parser.add_argument('--left_context_width', type=int, default=5, help='input left_context_width-width')
basic_parser.add_argument('--right_context_width', type=int, default=5, help='input right_context_width')
basic_parser.add_argument('--feat_step', type=int, default=1, help='the step of the sampling the feats')
basic_parser.add_argument('--normalize_type', default=1, type=int, help='Normalize type for training (1: dataset_cmvn, 2:utt_cmvn)')
basic_parser.add_argument('--num_utt_cmvn', default=30000, type=int, help='Number of utts for cmvn')
basic_parser.add_argument('--cmvn_file', default = 'cmvn.npy', type = str, help='File to cmvn')

basic_parser.add_argument('--use_data_balance', default=0, type=int, help='use date_rate to balance)')
basic_parser.add_argument('--num_utt_data_rate', default=30000, type=int, help='Number of utts for cmvn')
basic_parser.add_argument('--data_rate_file', default = None, type = str, help='File to cmvn')
basic_parser.add_argument('--pos_rate_thresh', default=0.0, type=float, help='the thresh to decide the block being positive')

basic_parser.add_argument('--specially_speech_rate', type=int, default=0, help='speech_rate 5')
basic_parser.add_argument('--specially_noisy_rate', type=int, default=0, help='noisy_rate 1')
basic_parser.add_argument('--specially_noise_rate', type=int, default=0, help='noise_rate 1')

basic_parser.add_argument('--speech_rate', type=int, default=12, help='speech_rate 5')
basic_parser.add_argument('--noisy_rate', type=int, default=0, help='noisy_rate 1')
basic_parser.add_argument('--noise_rate', type=int, default=4, help='noise_rate 1')
basic_parser.add_argument('--babble_rate', type=int, default=0, help='babble_rate 1')
basic_parser.add_argument('--music_rate', type=int, default=4, help='music_rate')

# traing configure
basic_parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
basic_parser.add_argument('--manual_seed', default = None, type=int, help='manual seed')

basic_parser.add_argument('--initialized_by_continue', dest='initialized_by_continue', type=int, default=0, help='1: initialized_by_continue, 0: no')
basic_parser.add_argument('--continue_from_name', default=None, type=str, help='best None model_best Continue from checkpoint model')

basic_parser.add_argument('--save_freq_steps', type=int, default=10000, help='frequency of saving the latest results')
basic_parser.add_argument('--save_by_steps', action='store_true', help='whether saves model by iteration')
basic_parser.add_argument('--print_freq_steps', type=int, default=100, help='the frequency of info printing, how many steps the training scans batch')
basic_parser.add_argument('--validate_freq_steps', type=int, default=20000, help='the frequency of validation, how many steps the training scans batch data')

# optimizer
basic_parser.add_argument('--epochs', default=30, type=int, help='Number of maximum epochs')
basic_parser.add_argument('--opt_type', type=str, default='adam', help='learning rate policy. [adadelta | adam | SGD]')
basic_parser.add_argument('--lr_policy', type=str, default='warmup', help='learning rate policy. [linear | step | plateau | cosine | warmup]')
basic_parser.add_argument('--lr', type=float, default=0.000001, help='initial learning rate for adam') 
basic_parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
basic_parser.add_argument('--max_norm', default=1000, type=int, help='Norm cutoff to prevent explosion of gradients')

basic_parser.add_argument('--lr_freq_steps', type=int, default=1, help='frequency of update learning rate')
basic_parser.add_argument('--steps', type=int, default=0, help='the steps for model training')

# for warmup
basic_parser.add_argument('--lr_factor_freq_step', default=2000000, type=int, help='the frequence step of reducing the lr_factor')
basic_parser.add_argument('--lr_factor', default=1.5, type=float, help='default = 1.0 tunable scalar multiply to learning rate')
basic_parser.add_argument('--warmup_step', default=10000, type=int, help='warmup steps')
# for linear and cosine
basic_parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
basic_parser.add_argument('--niter_decay', type=int, default=5000, help='# of iter to linearly decay learning rate to zero')
# for step
basic_parser.add_argument('--lr_decay_iters', type=int, default=2, help='multiply by a gamma every lr_decay_iters iterations')
# for plateau
basic_parser.add_argument('--lr_reduce_factor', default=0.9, type=float, help='lr_reduce_factor')
basic_parser.add_argument('--lr_reduce_threshold', default=1e-4, type=float, help='lr_reduce_threshold')
basic_parser.add_argument('--step_patience', default=1, type=int, help='step_patience')
basic_parser.add_argument('--min_lr', type=float, default=0.0000001, help=' min learning rate, default=0.00001')

# Loss
basic_parser.add_argument('--in_loss_type', type=str, default=None, help='loss_type, FocalLoss | CrossEntropyLoss | MarginLoss | SoftmaxMarginLoss') 
basic_parser.add_argument('--in_label_smoothing', default=0.0, type=float, help='label smoothing')
basic_parser.add_argument('--in_loss_weight', default=0.0, type=float, help='label smoothing')
basic_parser.add_argument('--in_out_act_type', type=str, default=None, help="Type of the activation function. None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu are supported")
basic_parser.add_argument('--out_loss_type', type=str, default='CrossEntropyLoss', help='loss_type, FocalLoss | CrossEntropyLoss | MarginLoss | SoftmaxMarginLoss') 
basic_parser.add_argument('--out_label_smoothing', default=0.1, type=float, help='label smoothing')
basic_parser.add_argument('--out_loss_weight', default=1.0, type=float, help='label smoothing')
basic_parser.add_argument('--out_out_act_type', type=str, default=None, help="Type of the activation function. None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu are supported")
basic_parser.add_argument('--num_classes', type=int, default=2, help='num_classes')
basic_parser.add_argument('--bias', action='store_true', help='whether to add bias')

# display configure
basic_parser.add_argument('--visdom_lr', dest='visdom_lr', type=int, default=1, help='Turn on visdom graphing learning rate')
basic_parser.add_argument('--visdom', dest='visdom', type=int, default=1, help='Turn on visdom graphing')
basic_parser.add_argument('--visdom_id', default='Noisy_VAD_Small', help='Identifier for visdom run')
basic_parser.add_argument('--display_server', type=str, default="http://10.20.6.175", help='visdom server of the web display')
basic_parser.add_argument('--display_port', type=int, default=8097, help=' 8097 visdom port of the web display')
basic_parser.add_argument('--visdom_freq_steps', type=int, default=1000, help='frequency of display learning rate')

# model
basic_parser.add_argument('--model', type=str, default='DeepFFAM', help='chooses which model to use. [DeepFFAM | DeepFFTransformerAM | DeepFFTransformerDeepFFAM]')
basic_parser.add_argument('--isTrain', action='store_true', help='whether train the model')
basic_parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
basic_parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

# additional parameters
basic_parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
basic_args = basic_parser.parse_args()

# configure for In DeepFF
in_deepff_parser = argparse.ArgumentParser(description='In DeepFF')
in_deepff_parser.add_argument('--hlayer_size', default='32-32-32', type=str, help='Hidden size of In DeepFF')
in_deepff_parser.add_argument('--hidden_act_type', type=str, default='relu', help="Type of the activation function. relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu are supported")
in_deepff_parser.add_argument('--out_act_type', type=str, default='relu', help="Type of the activation function. None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu are supported")
in_deepff_parser.add_argument('--batch_norm', action='store_true', help='whether to perform LayerNorm')
in_deepff_parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
in_deepff_parser.add_argument('--bias', action='store_true', help='whether to add bias')
in_deepff_args = in_deepff_parser.parse_args()

# configure for Transformer
transformer_parser = argparse.ArgumentParser(description='Transformer')
transformer_parser.add_argument('--n_layers', default=2, type=int, help='Number of layers')
transformer_parser.add_argument('--d_model', default=64, type=int, help='Dimension of model')
transformer_parser.add_argument('--d_inner', default=128, type=int, help='Dimension of inner')
transformer_parser.add_argument('--n_head', default=6, type=int, help='Number of Multi Head Attention (MHA)')
transformer_parser.add_argument('--d_k', default=32, type=int, help='Dimension of key')
transformer_parser.add_argument('--d_v', default=32, type=int, help='Dimension of value')
transformer_parser.add_argument('--batch_norm', action='store_true', help='whether to perform LayerNorm')
transformer_parser.add_argument('--residual_op', action='store_true', help='whether to perform Residual')
transformer_parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
transformer_args = transformer_parser.parse_args()

# configure for Out DeepFF
out_deepff_parser = argparse.ArgumentParser(description='Out DeepFF')
out_deepff_parser.add_argument('--hlayer_size', default='64-64', type=str, help='Hidden size of In DeepFF')
out_deepff_parser.add_argument('--hidden_act_type', type=str, default='relu', help="Type of the activation function. relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu are supported")
out_deepff_parser.add_argument('--out_act_type', type=str, default=None, help="Type of the activation function. None|relu|sigmoid|softmax|tanh|softplus|prelu|leakyrelu are supported")
out_deepff_parser.add_argument('--batch_norm', action='store_true', help='whether to perform LayerNorm')
out_deepff_parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
out_deepff_parser.add_argument('--bias', action='store_true', help='whether to add bias')
out_deepff_args = out_deepff_parser.parse_args()

''' For Debug '''
basic_args.verbose = True
basic_args.isTrain = True
basic_args.bias = True
''' For Debug '''
print(basic_args)

#CUDA_VISIBLE_DEVICES=2 python3 train_deepffam_small.py
## prepara dir for training ##
if not os.path.isdir(basic_args.works_dir):
    try:
        os.makedirs(basic_args.works_dir)
    except OSError:
        exit("ERROR: %s is not a dir" % (basic_args.works_dir))

basic_args.exp_path = os.path.join(basic_args.works_dir, 'exp')
if not os.path.isdir(basic_args.exp_path):
    os.makedirs(basic_args.exp_path)

if basic_args.cmvn_file is not None:
    basic_args.cmvn_file = os.path.join(basic_args.exp_path, basic_args.cmvn_file)

if basic_args.model.lower() == 'deepffam':  #DeepFFAM | DeepFFTransformerAM | DeepFFTransformerDeepFFAM
    # DeepFFAM-128-128-128-128
    in_size = (basic_args.left_context_width + basic_args.right_context_width + 1) * 40
    model_name = "%s-deepff-%d-%s-%d" % (basic_args.work_name, in_size, in_deepff_args.hlayer_size, basic_args.num_classes)
elif basic_args.model.lower() == 'deepfftransformeram':
    model_name = "%s-deepff-%s-transformer-%d" % (basic_args.work_name, in_deepff_args.hlayer_size, transformer_args.n_layers)
else:
    model_name = "%s-deepff-%s-transformer-%d-deepff-%s" % (basic_args.work_name, in_deepff_args.hlayer_size, transformer_args.n_layers, out_deepff_args.hlayer_size)
basic_args.model_name = "%s_%s" % (model_name, basic_args.out_loss_type)
if basic_args.in_loss_type is not None:
    basic_args.model_name = "%s_%s" % (basic_args.model_name, basic_args.in_loss_type)
basic_args.model_name = "%s_step-%d" % (basic_args.model_name, basic_args.feat_step)

#################### debug ###########################
#basic_args.model_name = 'CChip_KWS_DeepFFAM_FocalLoss'
#################### debug ###########################

basic_args.log_dir = os.path.join(basic_args.exp_path, basic_args.model_name)
if not os.path.exists(basic_args.log_dir):
    os.makedirs(basic_args.log_dir)

basic_args.model_dir = os.path.join(basic_args.exp_path, basic_args.model_name)
if not os.path.exists(basic_args.model_dir):
    os.makedirs(basic_args.model_dir)

## set seed ##
if basic_args.manual_seed is None:
    basic_args.manual_seed = random.randint(1, 10000)
print('manual_seed = %d' % basic_args.manual_seed)
random.seed(basic_args.manual_seed)
torch.manual_seed(basic_args.manual_seed)

## set gpu ##
str_ids = basic_args.gpu_ids.split(',')
basic_args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        basic_args.gpu_ids.append(id)
basic_args.device = torch.device("cuda:{}".format(basic_args.gpu_ids[0]) if len(basic_args.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")

## Data Prepare ##
print("num_workers = %d" % (basic_args.num_workers))
train_dataset = BlockDataset(data_args = basic_args, speech_dir = basic_args.speech_dir, noisy_dir = basic_args.noisy_dir, noise_dir = basic_args.noise_dir, 
                                dataset = 'train', cmvn_file = basic_args.cmvn_file, shuffle = False)
train_loader = BlockDataLoader(train_dataset, batch_size = basic_args.num_utt_per_loading, num_workers=basic_args.num_workers, shuffle=True, pin_memory=True)
                                            
val_dataset = BlockDataset(data_args = basic_args, speech_dir = basic_args.speech_dir, noisy_dir = basic_args.noisy_dir, noise_dir = basic_args.noise_dir, 
                                dataset = 'dev', cmvn_file = basic_args.cmvn_file, shuffle = False)
val_loader = BlockDataLoader(val_dataset, batch_size = basic_args.num_utt_per_loading, num_workers=basic_args.num_workers, shuffle=False, pin_memory=True)

feat_size              = train_dataset.feat_size
input_size             = train_dataset.in_size
if train_dataset.data_rate is not None:
    basic_args.data_rate   = torch.from_numpy(1.0 - train_dataset.data_rate)
    print(basic_args.data_rate)
else:
    basic_args.data_rate   = None

# configure for Output Layer
if basic_args.in_loss_type is not None:
    if basic_args.in_label_smoothing > 0.0:
        basic_args.in_loss_type = 'CrossEntropyLoss'

    if basic_args.in_loss_type.lower() == 'softmaxmarginloss':
        basic_args.in_out_act_type = 'sigmoid'
    else:
        basic_args.in_out_act_type = None

if basic_args.out_label_smoothing > 0.0:
    basic_args.out_loss_type = 'CrossEntropyLoss'

if basic_args.out_loss_type.lower() == 'softmaxmarginloss':
    basic_args.out_out_act_type = 'sigmoid'
else:
    basic_args.out_out_act_type = None
basic_args.bias = True

# configure for In DeepFF
hlayer_size = in_deepff_args.hlayer_size.split('-')
in_deepff_args.hlayer_size = []
for i in range(len(hlayer_size)):
    in_deepff_args.hlayer_size.append(int(hlayer_size[i]))
in_deepff_args.layer_size = []
in_deepff_args.layer_size.append(input_size)
in_deepff_args.layer_size.extend(in_deepff_args.hlayer_size)
in_deepff_args.batch_norm = False
in_deepff_args.bias = True

# configure for Transformer
transformer_args.d_model = in_deepff_args.layer_size[-1]
transformer_args.batch_norm = True
transformer_args.residual_op = True

# configure for Out DeepFF
hlayer_size = out_deepff_args.hlayer_size.split('-')
out_deepff_args.hlayer_size = []
for i in range(len(hlayer_size)):
    out_deepff_args.hlayer_size.append(int(hlayer_size[i]))
out_deepff_args.layer_size = []
out_deepff_args.layer_size.append(transformer_args.d_model)
out_deepff_args.layer_size.extend(out_deepff_args.hlayer_size)
out_deepff_args.batch_norm = True
out_deepff_args.bias = True

## AM Model Prepare ##
if basic_args.continue_from_name is None:
    ## AM Model Selection ##
    if basic_args.model.lower() == 'deepffam':  #DeepFFAM | DeepFFTransformerAM | DeepFFTransformerDeepFFAM
        basic_args.d_model = np.sum(np.array(in_deepff_args.layer_size)) / len(in_deepff_args.layer_size)
        am_model = DeepFFAM(basic_args, in_deepff_args)
    elif basic_args.model.lower() == 'deepfftransformeram':
        basic_args.d_model = transformer_args.d_model
        am_model = DeepFFTransformerAM(basic_args, in_deepff_args, transformer_args)
    else:
        basic_args.d_model = transformer_args.d_model
        am_model = DeepFFTransformerDeepFFAM(basic_args, in_deepff_args, transformer_args, out_deepff_args)

    am_model_state = None
else:
    if basic_args.initialized_by_continue == 1:
        ## AM Model Selection ##
        if basic_args.model.lower() == 'deepffam':  #DeepFFAM | DeepFFTransformerAM | DeepFFTransformerDeepFFAM
            basic_args.d_model = np.sum(np.array(in_deepff_args.layer_size)) / len(in_deepff_args.layer_size)
            am_model = DeepFFAM(basic_args, in_deepff_args)
        elif basic_args.model.lower() == 'deepfftransformeram':
            basic_args.d_model = transformer_args.d_model
            am_model = DeepFFTransformerAM(basic_args, in_deepff_args, transformer_args)
        else:
            basic_args.d_model = transformer_args.d_model
            am_model = DeepFFTransformerDeepFFAM(basic_args, in_deepff_args, transformer_args, out_deepff_args)

        am_model.init_model(model_path = basic_args.model_dir, continue_from_name = basic_args.continue_from_name)
        am_model_state = None
    else:
        ## AM Model Selection ##
        if basic_args.model.lower() == 'deepffam':  #DeepFFAM | DeepFFTransformerAM | DeepFFTransformerDeepFFAM
            am_model, am_model_state, basic_args, in_deepff_args = DeepFFAM.load_model(model_path = basic_args.model_dir, continue_from_name = basic_args.continue_from_name,
                                                                                                        given_model_configure = basic_args, gpu_ids = basic_args.gpu_ids, isTrain = basic_args.isTrain)
        elif basic_args.model.lower() == 'deepfftransformeram':
            am_model, am_model_state, basic_args, in_deepff_args, transformer_args = DeepFFTransformerAM.load_model(model_path = basic_args.model_dir, continue_from_name = basic_args.continue_from_name, 
                                                                                                        given_model_configure = basic_args, gpu_ids = basic_args.gpu_ids, isTrain = basic_args.isTrain)
        else:
            am_model, am_model_state, basic_args, in_deepff_args, transformer_args, out_deepff_args = DeepFFTransformerDeepFFAM.load_model(model_path = basic_args.model_dir, continue_from_name = basic_args.continue_from_name, 
                                                                                                        given_model_configure = basic_args, gpu_ids = basic_args.gpu_ids, isTrain = basic_args.isTrain)

print("Start: Awakeup AM Training %s" % basic_args.model_name)
basic_args.save_by_steps = False
am_trainer = AMTrainer(basic_args, train_loader, val_loader, am_model, am_model_state)
am_model = am_trainer.train(basic_args.epochs)
print("End: Awakeup AM Training %s" % basic_args.model_name)
