from __future__ import division
import time
import os
import shutil
import errno
from utils.utils import AverageMeter
import torch
from torch.autograd import Variable
import numpy as np
from utils.file_logger import FileLogger
from utils.plot import PlotFigure
from utils.utils import quantize_model
from utils import utils
import sys
from model.model import DeepFFAM, DeepFFTransformerAM, DeepFFTransformerDeepFFAM
from data.data_new_new import BlockDataset, BlockDataLoader, BlockBatchGenerator
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
from contextlib import contextmanager

@contextmanager
def printoptions(*args, **kwargs):
    original_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original_options)

class AMTrainer(object):
    def __init__(self, args, train_loader, val_loader, am_model, am_model_state = None):
        self.args           = args
        
        if hasattr(args, 'num_models'):
            self.num_models = args.num_models
        else:
            self.num_models = 1
        self.num_save_model = 0
        
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.am_model       = am_model

        if am_model_state is None:
            self.epoch              = 0
            self.tr_steps           = 0
            self.val_steps          = 0
            self.train_loss_all     = []
            self.train_acc_all      = []
            self.val_loss_all       = []
            self.val_acc_all        = []
            self.best_perform_acc   = 0.0
        else:
            self.epoch          = am_model_state['epoch']
            self.tr_steps       = am_model_state['tr_steps']
            self.val_steps      = am_model_state['val_steps']
            self.train_loss_all = am_model_state['tr_loss']
            self.train_acc_all  = am_model_state['tr_acc']
            self.val_loss_all   = am_model_state['val_loss']
            self.val_acc_all    = am_model_state['val_acc']
            self.best_perform_acc = self.val_acc_all[-1]
        
        if hasattr(args, 'lr_factor_freq_step'):
            self.lr_factor_freq_step = args.lr_factor_freq_step
        else:
            self.lr_factor_freq_step = 1000000
        self.lr_factor = args.lr_factor
                
        self.log_dir        = args.log_dir
        self.pos_batch_size     = args.pos_batch_size
        self.neg_batch_size     = args.neg_batch_size

        self.print_freq_steps       = args.print_freq_steps
        self.validate_freq_steps    = args.validate_freq_steps
        self.save_freq_steps        = args.save_freq_steps
        self.save_by_steps          = args.save_by_steps
        
        self.lr_freq_steps          = args.lr_freq_steps
        
        header              = ['tr_steps', 'val_steps', 'tr_loss',' tr_acc', 'val_loss', 'val_acc', 'in_tr_loss','in_tr_acc', 'in_val_loss', 'in_val_acc', 'tr_speed']
        self.file_logger    = FileLogger( os.path.join(self.log_dir, "out.tsv"), header )

        self.visdom         = args.visdom
        self.visdom_id      = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(server=args.display_server, port=args.display_port, env=args.visdom_id, raise_exceptions=True)
            
            self.tr_loss_vis_opts = dict(title="Train Loss", ylabel='Loss', xlabel='steps', legend=['train loss', 'val loss'])
            self.tr_loss_vis_window = None

            self.tr_acc_vis_opts = dict(title="Train Accurcy", ylabel='Accurcy', xlabel='steps', legend=['train acc', 'val acc'])
            self.tr_acc_vis_window = None

            self.val_loss_vis_opts = dict(title="Val Loss", ylabel='Loss', xlabel='steps', legend=['train loss', 'val loss'])
            self.val_loss_vis_window = None

            self.val_acc_vis_opts = dict(title="Val Accurcy", ylabel='Accurcy', xlabel='steps', legend=['train acc', 'val acc'])
            self.val_acc_vis_window = None

    def validate(self, val_loader):
        
        batch_val_acc   = AverageMeter()
        batch_val_loss   = AverageMeter()

        batch_val_acc_in   = AverageMeter()
        batch_val_loss_in   = AverageMeter()
        
        self.am_model.eval()

        valid_enum = tqdm(val_loader, desc='Valid')
        for i, (data) in enumerate(valid_enum, start=0):
            if data is None:
                continue
            
            batch_generator = BlockBatchGenerator(data = data)
            while not batch_generator.is_empty():
                
                #padded_input, target, input_lengths = batch_generator.next_batch_val(self.pos_batch_size, self.neg_batch_size)
                padded_input, target, input_lengths = batch_generator.next_batch(self.pos_batch_size, self.neg_batch_size)
                if padded_input is None or target is None or input_lengths is None:
                    print("Get a bad data!")
                    continue

                self.am_model.set_input(padded_input, target, input_lengths)
                self.am_model.test()
                losses = self.am_model.get_current_losses()
                if 'E' in self.am_model.loss_names:
                    batch_val_loss.update(float(losses['E']))
                if 'acc' in self.am_model.loss_names:
                    batch_val_acc.update(float(losses['acc']))
                
                if 'in_E' in self.am_model.loss_names:
                    batch_val_loss_in.update(float(losses['in_E']))
                if 'in_acc' in self.am_model.loss_names:
                    batch_val_acc_in.update(float(losses['in_acc']))
            
            #del batch_generator
            #gc.collect()

        print(' >> Validate: avg_loss = {0}, avg_acc = {1}, in_avg_loss = {2}, in_avg_acc = {3}'.format(
                                                                                        batch_val_loss.avg, batch_val_acc.avg, batch_val_loss_in.avg, batch_val_acc_in.avg))
        self.am_model.train()
        return batch_val_loss.avg, batch_val_acc.avg, batch_val_loss_in.avg, batch_val_acc_in.avg

    def train(self, epochs):

        val_loss, val_acc, in_val_loss, in_val_acc = self.validate(self.val_loader)
        self.best_perform_acc = val_acc
        print('>>Start performance: val_loss = {0}, val_acc = {1}, in_val_loss = {2}, in_val_acc = {3}<<'.format(val_loss, val_acc, in_val_loss, in_val_acc))

        self.am_model.train()

        val_freq_cost_time  = AverageMeter()
        val_freq_train_acc  = AverageMeter()
        val_freq_train_loss = AverageMeter()
        in_val_freq_train_acc  = AverageMeter()
        in_val_freq_train_loss = AverageMeter()

        print_freq_cost_time  = AverageMeter()
        print_freq_train_acc  = AverageMeter()
        print_freq_train_loss = AverageMeter()
        in_print_freq_train_acc  = AverageMeter()
        in_print_freq_train_loss = AverageMeter()

        start_epoch = self.epoch
        end_epoch   = start_epoch + epochs
        for epoch in range(start_epoch, end_epoch):
            self.epoch = epoch
            for i, (data) in enumerate(self.train_loader, start = 0):
                if data is None:
                    continue
                batch_generator = BlockBatchGenerator(data = data)
                while batch_generator.is_availabel() and not batch_generator.is_empty():
                    iter_start_time = time.time()

                    padded_input, target, input_lengths = batch_generator.next_batch(self.pos_batch_size, self.neg_batch_size)                
                    if padded_input is None or target is None or input_lengths is None:
                        print("Get a bad data!")
                        continue
                    self.am_model.set_input(padded_input, target, input_lengths)
                    self.am_model.optimize_parameters()
                    self.tr_steps = self.tr_steps + 1
                    
                    if self.tr_steps % self.lr_freq_steps == 0 and self.args.opt_type != 'adadelta':
                        self.am_model.update_learning_rate()
                    
                    if self.tr_steps % self.lr_factor_freq_step == 0:
                        self.lr_factor = 0.5 * self.lr_factor
                        self.am_model.set_lr_factor(self.lr_factor)
                    
                    losses = self.am_model.get_current_losses()
                    if 'E' in self.am_model.loss_names:
                        print_freq_train_loss.update(float(losses['E']))
                        val_freq_train_loss.update(float(losses['E']))
                    if 'acc' in self.am_model.loss_names:
                        print_freq_train_acc.update(float(losses['acc']))
                        val_freq_train_acc.update(float(losses['acc']))
                    if 'in_E' in self.am_model.loss_names:
                        in_print_freq_train_loss.update(float(losses['in_E']))
                        in_val_freq_train_loss.update(float(losses['in_E']))
                    if 'in_acc' in self.am_model.loss_names:
                        in_print_freq_train_acc.update(float(losses['in_acc']))
                        in_val_freq_train_acc.update(float(losses['in_acc']))

                    batch_freq_time = (time.time() - iter_start_time)
                    print_freq_cost_time.update(float(batch_freq_time))
                    val_freq_cost_time.update(float(batch_freq_time))

                    if self.tr_steps % self.print_freq_steps == 0:
                        '''
                        print('epoch[{0}] | steps[{1}] | avg_tr_loss: {2:.4f} | avg_tr_acc {3:.3f} | cur_tr_loss: {4:.4f} | cur_tr_acc {5:.3f} | {6:.1f} ms/batch'.format(
                          epoch, self.tr_steps, print_freq_train_loss.avg, print_freq_train_acc.avg, 
                          float(losses['E']), float(losses['acc']), print_freq_cost_time.avg), flush = True)
                        '''
                        print('epoch[{0}] | steps[{1}] | tr_loss: {2:.4f} | tr_acc {3:.2f} | in_tr_loss: {4:.2f} | in_tr_acc {5:.2f} | {6:.4f} ms/batch'.format(
                          epoch, self.tr_steps, print_freq_train_loss.avg, print_freq_train_acc.avg, 
                          in_print_freq_train_loss.avg, in_print_freq_train_acc.avg, print_freq_cost_time.avg), flush = True)

                        self.train_loss_all.append(print_freq_train_loss.avg)
                        self.train_acc_all.append(print_freq_train_acc.avg)

                        if self.visdom:
                            x_axis = torch.arange(0, len(self.train_loss_all))
                            y_axis = torch.from_numpy(np.array(self.train_loss_all))
                            if self.tr_loss_vis_window is None:
                                self.tr_loss_vis_window = self.vis.line(X = x_axis, Y = y_axis, opts = self.tr_loss_vis_opts)
                            else:
                                self.vis.line(X = x_axis, Y = y_axis, win = self.tr_loss_vis_window, update = 'replace')
                            
                            x_axis = torch.arange(0, len(self.train_acc_all))
                            y_axis = torch.from_numpy(np.array(self.train_acc_all))
                            if self.tr_acc_vis_window is None:
                                self.tr_acc_vis_window = self.vis.line(X = x_axis, Y = y_axis, opts = self.tr_acc_vis_opts)
                            else:
                                self.vis.line(X = x_axis, Y = y_axis, win = self.tr_acc_vis_window, update = 'replace')

                        print_freq_train_loss.reset()
                        print_freq_train_acc.reset()
                        in_print_freq_train_loss.reset()
                        in_print_freq_train_acc.reset()
                        print_freq_cost_time.reset()

                    if self.tr_steps % self.validate_freq_steps == 0:
                        val_loss, val_acc, in_val_loss, in_val_acc = self.validate(self.val_loader)
                        self.val_steps = self.val_steps + 1
                        
                        self.file_logger.write([self.tr_steps, self.val_steps, 
                                                val_freq_train_loss.avg, val_freq_train_acc.avg, val_loss, val_acc, 
                                                in_val_freq_train_loss.avg, in_val_freq_train_acc.avg, in_val_loss, in_val_acc,
                                                val_freq_cost_time.avg])
                        val_freq_train_loss.reset()
                        val_freq_train_acc.reset()
                        in_val_freq_train_loss.reset()
                        in_val_freq_train_acc.reset()
                        val_freq_cost_time.reset()

                        self.val_loss_all.append(val_loss)
                        self.val_acc_all.append(val_acc)
                        
                        if val_acc > self.best_perform_acc:
                            print("Found better validated model, saving to model_best.pth.tar")
                            self.best_perform_acc = val_acc
                            self.am_model.save_model('best', self.epoch, self.val_steps, self.train_loss_all, self.train_acc_all, self.val_loss_all, self.val_acc_all)
                        
                        if self.visdom:
                            x_axis = torch.arange(0, len(self.val_loss_all))
                            y_axis = torch.from_numpy(np.array(self.val_loss_all))
                            if self.val_loss_vis_window is None:
                                self.val_loss_vis_window = self.vis.line(X = x_axis, Y = y_axis, opts = self.val_loss_vis_opts)
                            else:
                                self.vis.line(X = x_axis, Y = y_axis, win = self.val_loss_vis_window, update = 'replace')
                            
                            x_axis = torch.arange(0, len(self.val_acc_all))
                            y_axis = torch.from_numpy(np.array(self.val_acc_all))
                            if self.val_acc_vis_window is None:
                                self.val_acc_vis_window = self.vis.line(X = x_axis, Y = y_axis, opts = self.val_acc_vis_opts)
                            else:
                                self.vis.line(X = x_axis, Y = y_axis, win = self.val_acc_vis_window, update = 'replace')

                    if self.tr_steps % self.save_freq_steps == 0:
                        print('saving the latest model (epoch %d, total_iters %d)' % (self.epoch, self.tr_steps))
                        self.num_save_model = self.num_save_model + 1
                        
                        save_model_id = self.num_save_model % self.num_models
                        save_suffix = '%d' % self.tr_steps if self.save_by_steps else 'latest_%d' % (save_model_id)
                        self.am_model.save_model(save_suffix, self.epoch, self.tr_steps, self.train_loss_all, self.train_acc_all, self.val_loss_all, self.val_acc_all)
                        
                #del batch_generator
                #gc.collect()

        print('#####################################################')
        val_loss, val_acc, in_val_loss, in_val_acc = self.validate(self.val_loader)
        print('Finished training with epochs[{0}] steps{1}, val_loss = {}, val_acc = {}, in_val_loss = {}, in_val_acc = {}'.format(self.epoch, self.tr_steps, 
                                                                                                                        val_loss, val_acc, in_val_loss, in_val_acc))
        print('#####################################################')
        
        '''
        plt.close()
        fig = plt.figure(figsize=(6, 6))
        axes1 = fig.add_subplot(111)
        plt.ion()
        fig.subplots_adjust(top=0.98, bottom=0.02, left=0.1, right=0.95, hspace=0.05, wspace=0.05)

        axes1.plot(np.array(self.train_acc_all))
        axes1.plot(np.array(self.val_acc_all))

        if self.args.quantize: 
            plt.savefig(os.path.join(self.model_dir, 'fixed_train.png'))
        else:
            plt.savefig(os.path.join(self.model_dir, 'train.png'))
        '''
        return self.am_model