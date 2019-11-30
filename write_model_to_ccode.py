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
from utils.utils import AverageMeter

basic_parser = argparse.ArgumentParser(description='Awakeup AM Training')

basic_parser.add_argument('--work_name', default='wakeup_xiaodu', type=str, help='the description to this work')
basic_parser.add_argument('--works_dir', metavar='DIR', help='the path to workspace dir', default="/home/snie/works/train_am/am_train_new/egs/cchip_kws/")

basic_parser.add_argument('--model', type=str, default='DeepFFAM', help='chooses which model to use. [DeepFFAM | DeepFFTransformerAM | DeepFFTransformerDeepFFAM]')
basic_parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

basic_parser.add_argument('--continue_from_name', default='best', type=str, help='model_best Continue from checkpoint model')
basic_parser.add_argument('--model_name', default='Universal_DeepFF_AM_CChip-deepff-440-512-256-256-256-256-218_CrossEntropyLoss_step-1', type=str, help='the model name for testing')
basic_parser.add_argument('--num_classes', type=int, default=218, help='num_classes')

basic_parser.add_argument('--isTrain', type=bool, default=True, help='Use binary NN or not, true for use, otherwise')

# data configure 
basic_parser.add_argument('--speech_dir', metavar='DIR', help='path to clean speech', default='/home/snie/works/train_am/am_train_new/egs/cchip_kws/geli_data/speech')
basic_parser.add_argument('--noisy_dir', metavar='DIR', help='path to noisy speech', default='/home/snie/works/train_am/am_train_new/egs/cchip_kws/geli_data/noisy')
basic_parser.add_argument('--noise_dir', metavar='DIR', help='path to noise', default='/home/snie/works/train_am/am_train_new/egs/cchip_kws/data/noise')

basic_parser.add_argument('--num_workers', type=int, default=2, help='the num_workers')
basic_parser.add_argument('--num_utt_per_loading', type=int, default=10, help='the number of utterances for each loading')
basic_parser.add_argument('--batch_size', type=int, default=512, help='the batch size for once training')

basic_parser.add_argument('--block_length', default=1, type=int, help='the size of block')
basic_parser.add_argument('--block_shift', default=1, type=int, help='the shift of block')

basic_parser.add_argument('--delta_order', default=0, type=int, help='Delta order for training')
basic_parser.add_argument('--left_context_width', type=int, default=5, help='input left_context_width-width')
basic_parser.add_argument('--right_context_width', type=int, default=5, help='input right_context_width')
basic_parser.add_argument('--feat_step', type=int, default=1, help='the step of the sampling the feats')     
basic_parser.add_argument('--normalize_type', default=1, type=int, help='Normalize type for training (1: dataset_cmvn, 2:utt_cmvn)')
basic_parser.add_argument('--num_utt_cmvn', default=30000, type=int, help='Number of utts for cmvn')
basic_parser.add_argument('--cmvn_file', default = 'cmvn.npy', type = str, help='File to cmvn')

basic_parser.add_argument('--speech_rate', type=int, default=5, help='the number of clean speech for one loading')
basic_parser.add_argument('--noisy_rate', type=int, default=25, help='the number of noisy speech for one loading')
basic_parser.add_argument('--noise_rate', type=int, default=2, help='the number of common noise for one loading')
basic_parser.add_argument('--babble_rate', type=int, default=2, help='the number of babble noise for one loading')
basic_parser.add_argument('--music_rate', type=int, default=2, help='the number of music noise for one loading')

basic_parser.add_argument('--manual_seed', default = 999, type=int, help='manual seed')

basic_args = basic_parser.parse_args()
args = basic_args

print("####################################################")
print("feat_step = %d" % (args.feat_step))
print("####################################################")

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

## prepara dir for training ##
if not os.path.isdir(basic_args.works_dir):
    try:
        os.makedirs(basic_args.works_dir)
    except OSError:
        exit("ERROR: %s is not a dir" % (basic_args.works_dir))

basic_args.exp_path = os.path.join(basic_args.works_dir, 'exp')
if not os.path.isdir(basic_args.exp_path):
    os.makedirs(basic_args.exp_path)

basic_args.model_dir = os.path.join(basic_args.exp_path, basic_args.model_name)
if not os.path.exists(basic_args.model_dir):
    os.makedirs(basic_args.model_dir)

if basic_args.cmvn_file is not None:
    basic_args.cmvn_file = os.path.join(basic_args.exp_path, basic_args.cmvn_file)

## Data Prepare ##
print("num_workers = %d" % (basic_args.num_workers))                                            
test_dataset = BlockDataset(data_args = basic_args, speech_dir = basic_args.speech_dir, noisy_dir = basic_args.noisy_dir, noise_dir = basic_args.noise_dir, 
                                dataset = 'test', cmvn_file = basic_args.cmvn_file, shuffle = False)
test_loader = BlockDataLoader(test_dataset, batch_size = basic_args.num_utt_per_loading, num_workers=basic_args.num_workers, shuffle=False, pin_memory=True)

feat_size = test_dataset.feat_size
input_size  = test_dataset.in_size

## AM Model Prepare ##
if basic_args.continue_from_name is None:
    exit("%s is not existed" % (os.path.join(basic_args.model_dir, basic_args.continue_from_name)))
else:
    if basic_args.model.lower() == 'deepffam':  #DeepFFAM | DeepFFTransformerAM | DeepFFTransformerDeepFFAM
        am_model, am_model_state, basic_args, in_deepff_args = DeepFFAM.load_model(model_path = basic_args.model_dir, continue_from_name = basic_args.continue_from_name, given_model_configure = basic_args, gpu_ids = basic_args.gpu_ids, isTrain = basic_args.isTrain, debug_quentize = True)
    elif basic_args.model.lower() == 'deepfftransformeram':
        am_model, am_model_state, basic_args, in_deepff_args, transformer_args = DeepFFTransformerAM.load_model(model_path = basic_args.model_dir, continue_from_name = basic_args.continue_from_name, given_model_configure = basic_args, gpu_ids = basic_args.gpu_ids, isTrain = basic_args.isTrain)
    else:
        am_model, am_model_state, basic_args, in_deepff_args, transformer_args, out_deepff_args = DeepFFTransformerDeepFFAM.load_model(model_path = basic_args.model_dir, continue_from_name = basic_args.continue_from_name, given_model_configure = basic_args, gpu_ids = basic_args.gpu_ids, isTrain = basic_args.isTrain)

batch_val_acc   = AverageMeter()
batch_val_loss   = AverageMeter()
batch_val_acc_in   = AverageMeter()
batch_val_loss_in   = AverageMeter()
am_model.eval()
valid_enum = tqdm(test_loader, desc='Valid')
for i, (data) in enumerate(valid_enum, start=0):
    if data is None:
        continue
    
    batch_generator = BlockBatchGenerator(data = data)
    while not batch_generator.is_empty():

        padded_input, target, input_lengths = batch_generator.next_batch(basic_args.batch_size, basic_args.batch_size)
        if padded_input is None or target is None or input_lengths is None:
            print("Get a bad data!")
            continue

        am_model.set_input(padded_input, target, input_lengths)
        am_model.test()
        losses = am_model.get_current_losses()
        if 'E' in am_model.loss_names:
            batch_val_loss.update(float(losses['E']))
        if 'acc' in am_model.loss_names:
            batch_val_acc.update(float(losses['acc']))
        
        if 'in_E' in am_model.loss_names:
            batch_val_loss_in.update(float(losses['in_E']))
        if 'in_acc' in am_model.loss_names:
            batch_val_acc_in.update(float(losses['in_acc']))

print(' >> Validate: avg_loss = {0}, avg_acc = {1}, in_avg_loss = {2}, in_avg_acc = {3}'.format(batch_val_loss.avg, batch_val_acc.avg, batch_val_loss_in.avg, batch_val_acc_in.avg))

am_model.print_statistical_information('statistics_model_best')

if basic_args.cmvn_file is not None:
    cmvn_file = os.path.join(basic_args.exp_path, basic_args.cmvn_file)
else:
    cmvn_file = os.path.join(basic_args.exp_path, 'cmvn.npy')

ccode_file      = os.path.join(basic_args.model_dir, 'skv_layers.cpp')
basic_file_name = 'basic_code.cc'
am_model.write_to_ccode(cmvn_file, ccode_file, basic_file_name, feat_size, input_size)