import torch
from torch import nn
from model.base_model import BaseModel, get_non_pad_mask, init_net, get_attn_pad_mask
import math
from model.networks import DeepFFNet, Outputer, TransformerFFLayer
from utils.utils import AverageMeter, to_np, accuracy, compute_acc
from model.loss import cal_performance, FocalLoss, IGNORE_ID
import os
import codecs
import torch.nn.functional as F


class DeepFFAM(BaseModel):
    def __init__(self, basic_args, deepff_args):
            """Initialize the DNN AM class.
            Parameters:
                basic_args: ()
                    basic_args.model_dir =

                deepff_args: (layer_size, hidden_act_type='relu', out_act_type = None, batch_norm = False, dropout = 0.0)
                    deepff_args.layer_size = [d_input, 128, 128, d_model]
                    deepff_args.hidden_act_type = 'relu'
                    deepff_args.out_act_type = 'relu'
                    deepff_args.batch_norm = True
                    deepff_args.dropout = 0.1
                    deepff_args.bias = True
                Loss:
                    basic_args.in_loss_type = None
                    basic_args.in_label_smoothing = 0.0
                    basic_args.in_loss_weight = 0.5
                    basic_args.in_out_act_type = None
                    basic_args.out_loss_type = 'CrossEntropyLoss'
                    basic_args.out_label_smoothing = 0.0
                    basic_args.out_loss_weight = 0.5
                    basic_args.out_out_act_type = None
                    basic_args.num_classes = 218
                    basic_args.bias = True
            """
            BaseModel.__init__(self, basic_args)

            if hasattr(basic_args, 'steps'):
                self.steps = basic_args.steps
            else:
                self.steps = 0

            self.basic_args  = basic_args
            self.deepff_args = deepff_args
            self.model_names = ['deepff', 'out_prj']

            # define and initialize the DeepFF Net
            self.deepff = DeepFFNet(layer_size = deepff_args.layer_size, binary = basic_args.binary, hidden_act_type = deepff_args.hidden_act_type,
                                    out_act_type = deepff_args.out_act_type, batch_norm = deepff_args.batch_norm,
                                    dropout = deepff_args.dropout, bias = deepff_args.bias)
            self.deepff = init_net(self.deepff, basic_args.init_type, basic_args.init_gain, self.device)

            # define and initialize the out_prj Net
            self.out_prj = Outputer(in_size = deepff_args.layer_size[-1], out_size = basic_args.num_classes,
                                    binary = basic_args.binary, out_act_type = basic_args.out_out_act_type,
                                    batch_norm = deepff_args.batch_norm, bias = basic_args.bias)
            self.out_prj = init_net(self.out_prj, basic_args.init_type, basic_args.init_gain, self.device)

            self.loss_names = ['E', 'acc']
            if basic_args.isTrain:
                # define loss functions of DNN AM
                if basic_args.out_loss_type.lower() == 'crossentropyloss':
                    self.criterion = torch.nn.CrossEntropyLoss(weight = basic_args.data_rate, ignore_index = IGNORE_ID).to(self.device)
                elif basic_args.out_loss_type.lower() == 'focalloss':
                    self.criterion = FocalLoss(basic_args.num_classes, self.device, ignore_index = IGNORE_ID, alpha = basic_args.data_rate, gamma = 1.0).to(self.device)
                elif basic_args.out_loss_type.lower() == 'marginloss':
                    self.criterion = torch.nn.MultiMarginLoss().to(self.device)
                elif basic_args.out_loss_type.lower() == 'softmaxmarginloss':
                    self.criterion = torch.nn.MultiMarginLoss().to(self.device)
                else:
                    raise Exception('{} train_type error'.format( basic_args.out_loss_type))

                # define the optimizer of DeepFF
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_deepff = torch.optim.Adadelta(self.deepff.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_deepff = torch.optim.Adam(self.deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_deepff = torch.optim.SGD(self.deepff.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_deepff = torch.optim.Adam(self.deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_deepff)

                # define the optimizer of out_prj
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_out = torch.optim.Adadelta(self.out_prj.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_out = torch.optim.Adam(self.out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_out = torch.optim.SGD(self.out_prj.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_out = torch.optim.Adam(self.out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_out)

            self.setup(basic_args)

    @classmethod
    def load_model(cls, model_path, continue_from_name = 'best', given_model_configure = None, gpu_ids = [1], isTrain = False, debug_quentize = False):

        if continue_from_name is None:
            exit("ERROR: continue_from_model is None")
        
        continue_name = continue_from_name.split('-')[0]
        load_filename = 'model_%s.configure' % (continue_name)
        configure_path = os.path.join(model_path, load_filename)
        if not os.path.exists(configure_path):
            exit("ERROR: %s is not existed" % (configure_path))

        model_configure = torch.load(configure_path, map_location=lambda storage, loc: storage)
        basic_args = model_configure['basic_args']
        basic_args.model_dir = model_path
        basic_args.continue_from_name = continue_from_name
        basic_args.gpu_ids = gpu_ids
        basic_args.isTrain = isTrain

        if given_model_configure is not None:
            if hasattr(given_model_configure, 'bias'):
                basic_args.bias = given_model_configure.bias
            
            if hasattr(given_model_configure, 'batch_size'):
                basic_args.batch_size = given_model_configure.batch_size
            
            if hasattr(given_model_configure, 'num_models'):
                basic_args.num_models = given_model_configure.num_models
            
            if hasattr(given_model_configure, 'log_dir'):
                basic_args.log_dir = given_model_configure.log_dir
            
            if hasattr(given_model_configure, 'data_rate'):
                basic_args.data_rate = given_model_configure.data_rate
            else:
                basic_args.data_rate = None

            if hasattr(given_model_configure, 'save_freq_steps'):
                basic_args.save_freq_steps = given_model_configure.save_freq_steps
            if hasattr(given_model_configure, 'save_by_steps'):
                basic_args.save_by_steps = given_model_configure.save_by_steps
            if hasattr(given_model_configure, 'print_freq_steps'):
                basic_args.print_freq_steps = given_model_configure.print_freq_steps
            if hasattr(given_model_configure, 'validate_freq_steps'):
                basic_args.validate_freq_steps = given_model_configure.validate_freq_steps

            if hasattr(given_model_configure, 'epochs'):
                basic_args.epochs = given_model_configure.epochs
            if hasattr(given_model_configure, 'opt_type'):
                basic_args.opt_type = given_model_configure.opt_type
            if hasattr(given_model_configure, 'lr_policy'):
                basic_args.lr_policy = given_model_configure.lr_policy
            if hasattr(given_model_configure, 'lr'):
                basic_args.lr = given_model_configure.lr
            if hasattr(given_model_configure, 'beta1'):
                basic_args.beta1 = given_model_configure.beta1
            if hasattr(given_model_configure, 'max_norm'):
                basic_args.max_norm = given_model_configure.max_norm

            if hasattr(given_model_configure, 'lr_freq_steps'):
                basic_args.lr_freq_steps = given_model_configure.lr_freq_steps

            if hasattr(given_model_configure, 'lr_factor_freq_step'):
                basic_args.lr_factor_freq_step = given_model_configure.lr_factor_freq_step
            if hasattr(given_model_configure, 'lr_factor'):
                basic_args.lr_factor = given_model_configure.lr_factor
            if hasattr(given_model_configure, 'warmup_step'):
                basic_args.warmup_step = given_model_configure.warmup_step

            if hasattr(given_model_configure, 'niter'):
                basic_args.niter = given_model_configure.niter
            if hasattr(given_model_configure, 'niter_decay'):
                basic_args.niter_decay = given_model_configure.niter_decay

            if hasattr(given_model_configure, 'lr_decay_iters'):
                basic_args.lr_decay_iters = given_model_configure.lr_decay_iters

            if hasattr(given_model_configure, 'lr_reduce_factor'):
                basic_args.lr_reduce_factor = given_model_configure.lr_reduce_factor
            if hasattr(given_model_configure, 'lr_reduce_threshold'):
                basic_args.lr_reduce_threshold = given_model_configure.lr_reduce_threshold
            if hasattr(given_model_configure, 'step_patience'):
                basic_args.step_patience = given_model_configure.step_patience
            if hasattr(given_model_configure, 'min_lr'):
                basic_args.min_lr = given_model_configure.min_lr

            if hasattr(given_model_configure, 'visdom_lr'):
                basic_args.visdom_lr = given_model_configure.visdom_lr
            if hasattr(given_model_configure, 'visdom'):
                basic_args.visdom = given_model_configure.visdom
            if hasattr(given_model_configure, 'visdom_id'):
                basic_args.visdom_id = given_model_configure.visdom_id
            if hasattr(given_model_configure, 'display_server'):
                basic_args.display_server = given_model_configure.display_server
            if hasattr(given_model_configure, 'display_port'):
                basic_args.display_port = given_model_configure.display_port
            if hasattr(given_model_configure, 'visdom_freq_steps'):
                basic_args.visdom_freq_steps = given_model_configure.visdom_freq_steps

            if hasattr(given_model_configure, 'verbose'):
                basic_args.verbose = given_model_configure.verbose

        deepff_args = model_configure['deepff_args']
        basic_args.steps = model_configure['tr_steps']
        
        from model.networks import set_debug_quentize
        set_debug_quentize(debug_quentize)

        model = cls(basic_args, deepff_args)
        model.load_networks(continue_from_name)
        model.steps =  model_configure['tr_steps']

        model_state = {
            'epoch': model_configure['epoch'],
            'tr_steps': model_configure['tr_steps'],
            'val_steps': model_configure['val_steps'],
            'tr_loss': model_configure['tr_loss'],
            'tr_acc': model_configure['tr_acc'],
            'val_loss': model_configure['val_loss'],
            'val_acc': model_configure['val_acc']
        }
        return model, model_state, basic_args, deepff_args

    def save_model(self, suffix_name, epoch, val_steps, tr_loss = None, tr_acc = None, val_loss = None, val_acc = None):
        configure_package = {
            'epoch': epoch,
            'tr_steps': self.steps,
            'val_steps': val_steps,
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'basic_args': self.basic_args,
            'deepff_args': self.deepff_args
        }
        save_filename = 'model_%s.configure' % (suffix_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(configure_package, save_path)
        self.save_networks(suffix_name)

    def set_input(self, padded_input, target = None, input_lengths = None):
        """
        Args:
            padded_input: (num_block, num_frame, d_x)
            target: (num_block, num_frame)
            input_lengths: (num_block, 1)
        Returns:
            enc_output: (num_block, num_frame, d_y)
        """
        self.steps = self.steps + 1

        self.padded_input = padded_input.to(self.device)

        if target is not None:
            self.target = target.to(self.device)
        else:
            self.target = None

        if input_lengths is not None:
            self.input_lengths = input_lengths
            self.non_pad_mask = get_non_pad_mask(self.padded_input, input_lengths = self.input_lengths)
        else:
            self.input_lengths = None
            self.non_pad_mask = None

    def inference(self, padded_input, input_lengths = None):

        self.padded_input = padded_input.to(self.device)

        if input_lengths is not None:
            self.input_lengths = input_lengths
            self.non_pad_mask = get_non_pad_mask(self.padded_input, input_lengths = self.input_lengths)
        else:
            self.input_lengths = None
            self.non_pad_mask = None

        # predict: (num_block, num_frame, num_phone)
        predict = self.out_prj(self.deepff(self.padded_input))

        predict_id = predict.max(2, keepdim=True)[1] # get the index of the max log-probability

        return predict, predict_id

    def forward(self):
        h = self.deepff(self.padded_input)
        self.predict = self.out_prj(h)

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.backward_E(perform_backward = False)

    def print_statistical_information(self, suffix_name = None):
        if suffix_name is not None:
            save_filename = '%s.txt' % (suffix_name)
            out_file_name = os.path.join(self.save_dir, save_filename)
            f = codecs.open(out_file_name, 'w', 'utf-8')
        else:
            f = None

        print("layes fc_min fc_max fc_mean fc_std bn_min bn_max bn_mean bn_std act_min act_max act_mean act_std")
        if f is not None:
            f.write("layes fc_min fc_max fc_mean fc_std bn_min bn_max bn_mean bn_std act_min act_max act_mean act_std\r\n")

        num_layer = len(self.deepff.NNet)
        for layer in range(num_layer):
            fc_min  = self.deepff.NNet[layer].fc_min
            fc_max  = self.deepff.NNet[layer].fc_max
            fc_mean = self.deepff.NNet[layer].fc_mean
            fc_std  = self.deepff.NNet[layer].fc_std

            bn_min  = self.deepff.NNet[layer].bn_min
            bn_max  = self.deepff.NNet[layer].bn_max
            bn_mean = self.deepff.NNet[layer].bn_mean
            bn_std  = self.deepff.NNet[layer].bn_std

            act_min  = self.deepff.NNet[layer].act_min
            act_max  = self.deepff.NNet[layer].act_max
            act_mean = self.deepff.NNet[layer].act_mean
            act_std  = self.deepff.NNet[layer].act_std

            print("layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n"
                % (layer, fc_min, fc_max, fc_mean, fc_std, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))
            if f is not None:
                f.write("layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n"
                      % (layer, fc_min, fc_max, fc_mean, fc_std, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))

        fc_min  = self.out_prj.out_proj[0].fc_min
        fc_max  = self.out_prj.out_proj[0].fc_max
        fc_mean = self.out_prj.out_proj[0].fc_mean
        fc_std  = self.out_prj.out_proj[0].fc_std

        bn_min  = self.out_prj.out_proj[0].bn_min
        bn_max  = self.out_prj.out_proj[0].bn_max
        bn_mean = self.out_prj.out_proj[0].bn_mean
        bn_std  = self.out_prj.out_proj[0].bn_std

        act_min  = self.out_prj.out_proj[0].act_min
        act_max  = self.out_prj.out_proj[0].act_max
        act_mean = self.out_prj.out_proj[0].act_mean
        act_std  = self.out_prj.out_proj[0].act_std

        print("layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n"
            % (layer, fc_min, fc_max, fc_mean, fc_std, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))
        if f is not None:
            f.write("layer_%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n"
                 % (layer, fc_min, fc_max, fc_mean, fc_std, bn_min, bn_max, bn_mean, bn_std, act_min, act_max, act_mean, act_std))

        if f is not None:
            f.close()

    def backward_E(self, perform_backward = True):

        self.loss_E, self.loss_acc = cal_performance(self.predict, self.target, self.criterion, self.basic_args.out_label_smoothing)
        if perform_backward:
            self.loss_E.backward()

    def zero_grad(self):
        self.optimizer_deepff.zero_grad()
        self.optimizer_out.zero_grad()

    def step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.deepff.parameters(), self.basic_args.max_norm)
        for p in list(self.deepff.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update deepff.')
        else:
            self.optimizer_deepff.step()
        for p in list(self.deepff.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))

        grad_norm = torch.nn.utils.clip_grad_norm_(self.out_prj.parameters(), self.basic_args.max_norm)
        for p in list(self.out_prj.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update out_prj.')
        else:
            self.optimizer_out.step()
        for p in list(self.out_prj.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))

    def optimize_parameters(self):

        # Perform forward
        self.forward()

        # Set zero_grad to the optimizer
        self.zero_grad()

        # Perform backward
        self.backward_E()

        # Update the weight
        self.step()



class DeepFFTransformerAM(BaseModel):
    def __init__(self, basic_args, deepff_args, transformer_args):
            """Initialize the DNN AM class.
            Parameters:
                basic_args: ()
                    basic_args.model_dir =

                deepff_args: (layer_size, hidden_act_type='relu', out_act_type = None, batch_norm = False, dropout = 0.0, name = 'DeepFFNet', version = '0.0.1')
                    deepff_args.layer_size = [d_input, 128, 128, d_model]
                    deepff_args.hidden_act_type = 'relu'
                    deepff_args.out_act_type = 'relu'
                    deepff_args.batch_norm = True
                    deepff_args.dropout = 0.1
                    deepff_args.bias = True
                    deepff_args.name = 'InDeepFFNet'
                    deepff_args.version = '0.0.1'

                transformer_args: ( d_model, d_inner, n_layers, n_head, d_k, d_v, residual_op = False, batch_norm = False, dropout = 0.0)
                    transformer_args.d_model = d_model
                    transformer_args.d_inner = d_inner
                    transformer_args.n_head = n_head
                    transformer_args.n_layers = n_layers
                    transformer_args.d_k = d_k
                    transformer_args.d_v = d_v
                    transformer_args.residual_op = True
                    transformer_args.batch_norm = True
                    transformer_args.dropout = 0.1

                Loss:
                    basic_args.in_loss_type = None
                    basic_args.in_label_smoothing = 0.0
                    basic_args.in_loss_weight = 0.5
                    basic_args.in_out_act_type = None
                    basic_args.out_loss_type = 'CrossEntropyLoss'
                    basic_args.out_label_smoothing = 0.0
                    basic_args.out_loss_weight = 0.5
                    basic_args.out_out_act_type = None
                    basic_args.num_classes = 218
                    basic_args.bias = True
            """
            BaseModel.__init__(self, basic_args)
            if hasattr(basic_args, 'steps'):
                self.steps = basic_args.steps
            else:
                self.steps = 0

            self.basic_args = basic_args
            self.deepff_args = deepff_args
            self.transformer_args = transformer_args

            if basic_args.in_loss_type is None:
                self.model_names = ['deepff', 'transformer', 'out_prj']
            else:
                self.model_names = ['deepff', 'transformer', 'out_prj', 'in_out_prj']

            # define and initialize the DeepFF Net
            self.deepff = DeepFFNet(layer_size = deepff_args.layer_size, hidden_act_type = deepff_args.hidden_act_type, 
                                    out_act_type = deepff_args.out_act_type, batch_norm = deepff_args.batch_norm,
                                    dropout = deepff_args.dropout, bias = deepff_args.bias)
            self.deepff = init_net(self.deepff, basic_args.init_type, basic_args.init_gain, self.device)

            # define and initialize the Transformer Net
            self.transformer = nn.ModuleList([
            TransformerFFLayer(in_size = transformer_args.d_model, d_model = transformer_args.d_model, d_ff_inner = transformer_args.d_inner, query_key_size = transformer_args.d_k, 
                                value_size = transformer_args.d_v, num_head = transformer_args.n_head, layer_norm = transformer_args.batch_norm, 
                                residual_op = transformer_args.residual_op, dropout = transformer_args.dropout)
            for _ in range(transformer_args.n_layers)])
            self.transformer = init_net(self.transformer, basic_args.init_type, basic_args.init_gain, self.device)

            # define and initialize the out_prj Net
            if basic_args.in_loss_type is not None:
                self.in_out_prj = Outputer(in_size = deepff_args.layer_size[-1], out_size = basic_args.num_classes, out_act_type = basic_args.in_out_act_type, bias = basic_args.bias)
                self.in_out_prj = init_net(self.in_out_prj, basic_args.init_type, basic_args.init_gain, self.device)
            else:
                 self.in_out_prj = None

            self.out_prj = Outputer(in_size = transformer_args.d_model, out_size = basic_args.num_classes, out_act_type = basic_args.out_out_act_type, bias = basic_args.bias)
            self.out_prj = init_net(self.out_prj, basic_args.init_type, basic_args.init_gain, self.device)

            if basic_args.in_loss_type is None:
                self.loss_names = ['E', 'acc']
            else:
                self.loss_names = ['E', 'acc', 'in_E', 'in_acc']

            if basic_args.isTrain:
                # define loss functions of DNN AM
                if basic_args.out_loss_type.lower() == 'crossentropyloss':
                    self.criterion = torch.nn.CrossEntropyLoss(weight = basic_args.data_rate, ignore_index = IGNORE_ID).to(self.device)
                elif basic_args.out_loss_type.lower() == 'focalloss':
                    self.criterion = FocalLoss(basic_args.num_classes, self.device, ignore_index = IGNORE_ID, alpha = basic_args.data_rate).to(self.device)
                elif basic_args.out_loss_type.lower() == 'marginloss':
                    self.criterion = torch.nn.MultiMarginLoss().to(self.device)
                elif basic_args.out_loss_type.lower() == 'softmaxmarginloss':
                    self.criterion = torch.nn.MultiMarginLoss().to(self.device)
                else:
                    raise Exception('{} train_type error'.format( basic_args.out_loss_type))

                if self.in_out_prj is not None:
                    if basic_args.in_loss_type.lower() == 'crossentropyloss':
                        self.in_criterion = torch.nn.CrossEntropyLoss(weight = basic_args.data_rate, ignore_index = IGNORE_ID).to(self.device)
                    elif basic_args.in_loss_type.lower() == 'focalloss':
                        self.in_criterion = FocalLoss(basic_args.num_classes, self.device, ignore_index = IGNORE_ID, alpha = basic_args.data_rate).to(self.device)
                    elif basic_args.in_loss_type.lower() == 'marginloss':
                        self.in_criterion = torch.nn.MultiMarginLoss().to(self.device)
                    elif basic_args.in_loss_type.lower() == 'softmaxmarginloss':
                        self.in_criterion = torch.nn.MultiMarginLoss().to(self.device)
                    else:
                        raise Exception('{} train_type error'.format( basic_args.in_loss_type))
                else:
                    self.in_criterion = None

                # define the optimizer of DeepFF
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_deepff = torch.optim.Adadelta(self.deepff.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_deepff = torch.optim.Adam(self.deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_deepff = torch.optim.SGD(self.deepff.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_deepff = torch.optim.Adam(self.deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_deepff)

                # define the optimizer of transformer
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_transformer = torch.optim.Adadelta(self.transformer.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_transformer = torch.optim.Adam(self.transformer.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_transformer = torch.optim.SGD(self.transformer.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_transformer = torch.optim.Adam(self.transformer.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_transformer)

                # define the optimizer of out_prj
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_out = torch.optim.Adadelta(self.out_prj.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_out = torch.optim.Adam(self.out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_out = torch.optim.SGD(self.out_prj.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_out = torch.optim.Adam(self.out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_out)

                if self.in_out_prj is not None:
                    if basic_args.opt_type.lower() == 'adadelta':
                        self.in_optimizer_out = torch.optim.Adadelta(self.in_out_prj.parameters(), rho=0.95, eps=1e-6)
                    elif basic_args.opt_type.lower() == 'adam':
                        self.in_optimizer_out = torch.optim.Adam(self.in_out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                    elif basic_args.opt_type.lower() == 'sgd':
                        self.in_optimizer_out = torch.optim.SGD(self.in_out_prj.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                    else:
                        self.in_optimizer_out = torch.optim.Adam(self.in_out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                    self.optimizers.append(self.in_optimizer_out)
                else:
                    self.in_optimizer_out = None

            self.setup(basic_args)

    @classmethod
    def load_model(cls, model_path, continue_from_name = 'best', given_model_configure = None, gpu_ids = [1], isTrain = False):
        if continue_from_name is None:
            exit("ERROR: continue_from_model is None")

        load_filename = 'model_%s.configure' % (continue_from_name)
        configure_path = os.path.join(model_path, load_filename)

        if not os.path.exists(configure_path):
            exit("ERROR: %s is not existed" % (configure_path))

        model_configure = torch.load(configure_path, map_location=lambda storage, loc: storage)

        basic_args = model_configure['basic_args']
        basic_args.model_dir = model_path
        basic_args.continue_from_name = continue_from_name
        basic_args.gpu_ids = gpu_ids
        basic_args.isTrain = isTrain

        if given_model_configure is not None:

            if hasattr(given_model_configure, 'data_rate'):
                basic_args.data_rate = given_model_configure.data_rate

            if hasattr(given_model_configure, 'save_freq_steps'):
                basic_args.save_freq_steps = given_model_configure.save_freq_steps
            if hasattr(given_model_configure, 'save_by_steps'):
                basic_args.save_by_steps = given_model_configure.save_by_steps
            if hasattr(given_model_configure, 'print_freq_steps'):
                basic_args.print_freq_steps = given_model_configure.print_freq_steps
            if hasattr(given_model_configure, 'validate_freq_steps'):
                basic_args.validate_freq_steps = given_model_configure.validate_freq_steps

            if hasattr(given_model_configure, 'epochs'):
                basic_args.epochs = given_model_configure.epochs
            if hasattr(given_model_configure, 'opt_type'):
                basic_args.opt_type = given_model_configure.opt_type
            if hasattr(given_model_configure, 'lr_policy'):
                basic_args.lr_policy = given_model_configure.lr_policy
            if hasattr(given_model_configure, 'lr'):
                basic_args.lr = given_model_configure.lr
            if hasattr(given_model_configure, 'beta1'):
                basic_args.beta1 = given_model_configure.beta1
            if hasattr(given_model_configure, 'max_norm'):
                basic_args.max_norm = given_model_configure.max_norm

            if hasattr(given_model_configure, 'lr_freq_steps'):
                basic_args.lr_freq_steps = given_model_configure.lr_freq_steps

            if hasattr(given_model_configure, 'lr_factor_freq_step'):
                basic_args.lr_factor_freq_step = given_model_configure.lr_factor_freq_step
            if hasattr(given_model_configure, 'lr_factor'):
                basic_args.lr_factor = given_model_configure.lr_factor
            if hasattr(given_model_configure, 'warmup_step'):
                basic_args.warmup_step = given_model_configure.warmup_step

            if hasattr(given_model_configure, 'niter'):
                basic_args.niter = given_model_configure.niter
            if hasattr(given_model_configure, 'niter_decay'):
                basic_args.niter_decay = given_model_configure.niter_decay

            if hasattr(given_model_configure, 'lr_decay_iters'):
                basic_args.lr_decay_iters = given_model_configure.lr_decay_iters

            if hasattr(given_model_configure, 'lr_reduce_factor'):
                basic_args.lr_reduce_factor = given_model_configure.lr_reduce_factor
            if hasattr(given_model_configure, 'lr_reduce_threshold'):
                basic_args.lr_reduce_threshold = given_model_configure.lr_reduce_threshold
            if hasattr(given_model_configure, 'step_patience'):
                basic_args.step_patience = given_model_configure.step_patience
            if hasattr(given_model_configure, 'min_lr'):
                basic_args.min_lr = given_model_configure.min_lr

            if hasattr(given_model_configure, 'visdom_lr'):
                basic_args.visdom_lr = given_model_configure.visdom_lr
            if hasattr(given_model_configure, 'visdom'):
                basic_args.visdom = given_model_configure.visdom
            if hasattr(given_model_configure, 'visdom_id'):
                basic_args.visdom_id = given_model_configure.visdom_id
            if hasattr(given_model_configure, 'display_server'):
                basic_args.display_server = given_model_configure.display_server
            if hasattr(given_model_configure, 'display_port'):
                basic_args.display_port = given_model_configure.display_port
            if hasattr(given_model_configure, 'visdom_freq_steps'):
                basic_args.visdom_freq_steps = given_model_configure.visdom_freq_steps

            if hasattr(given_model_configure, 'verbose'):
                basic_args.verbose = given_model_configure.verbose

        deepff_args = model_configure['deepff_args']
        transformer_args = model_configure['transformer_args']

        basic_args.steps = model_configure['tr_steps']
        model = cls(basic_args, deepff_args, transformer_args)
        model.load_networks(continue_from_name)
        model.steps = model_configure['tr_steps']

        model_state = {
            'epoch': model_configure['epoch'],
            'tr_steps': model_configure['tr_steps'],
            'val_steps': model_configure['val_steps'],
            'tr_loss': model_configure['tr_loss'],
            'tr_acc': model_configure['tr_acc'],
            'val_loss': model_configure['val_loss'],
            'val_acc': model_configure['val_acc']
        }

        return model, model_state, basic_args, deepff_args, transformer_args

    def save_model(self, suffix_name, epoch, val_steps, tr_loss = None, tr_acc = None, val_loss = None, val_acc = None):

        configure_package = {
            'epoch': epoch,
            'tr_steps': self.steps,
            'val_steps': val_steps,
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'basic_args': self.basic_args,
            'deepff_args': self.deepff_args,
            'transformer_args': self.transformer_args
        }
        save_filename = 'model_%s.configure' % (suffix_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(configure_package, save_path)
        self.save_networks(suffix_name)

    def set_input(self, padded_input, target = None, input_lengths = None):
        """
        Args:
            padded_input: (num_block, num_frame, d_x)
            target: (num_block, num_frame)
            input_lengths: (num_block, 1)
        Returns:
            enc_output: (num_block, num_frame, d_y)
        """
        self.steps = self.steps + 1

        self.padded_input = padded_input.to(self.device)

        if target is not None:
            self.target = target.to(self.device)
        else:
            self.target = None

        if input_lengths is not None:
            self.input_lengths = input_lengths
            self.non_pad_mask = get_non_pad_mask(self.padded_input, input_lengths = self.input_lengths)
            length = padded_input.size(1)
            self.slf_attn_mask = get_attn_pad_mask(self.padded_input, self.input_lengths, length)
        else:
            self.input_lengths = None
            self.non_pad_mask = None

    def inference(self, padded_input, input_lengths = None):

        self.padded_input = padded_input.to(self.device)

        if input_lengths is not None:
            self.input_lengths = input_lengths
            self.non_pad_mask = get_non_pad_mask(self.padded_input, input_lengths = self.input_lengths)
            length = padded_input.size(1)
            self.slf_attn_mask = get_attn_pad_mask(self.padded_input, self.input_lengths, length)
        else:
            self.input_lengths = None
            self.non_pad_mask = None
            self.slf_attn_mask = None

        # enc_output: (num_block, num_frame, num_phone)
        enc_output = self.deepff(self.padded_input)

        if self.in_out_prj is not None:
            in_predict = self.in_out_prj(enc_output)
            in_predict_id = in_predict.max(2, keepdim=True)[1]
        else:
            in_predict = None
            in_predict_id = None

        for transformer_layer in self.transformer:
            enc_output, _ = transformer_layer(enc_output, non_pad_mask = self.non_pad_mask, slf_attn_mask = self.slf_attn_mask)

        predict = self.out_prj(enc_output)

        predict_id = predict.max(2, keepdim=True)[1]

        return predict, predict_id, in_predict, in_predict_id

    def forward(self):

        # enc_output: (num_block, num_frame, num_phone)
        enc_output = self.deepff(self.padded_input)

        if self.in_out_prj is not None:
            self.in_predict = self.in_out_prj(enc_output)
        else:
            self.in_predict = None

        for transformer_layer in self.transformer:
            enc_output, _ = transformer_layer(enc_output, non_pad_mask = self.non_pad_mask, slf_attn_mask = self.slf_attn_mask)

        self.predict = self.out_prj(enc_output)

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.backward_E(perform_backward = False)

    def backward_E(self, perform_backward = True):
        self.loss_E, self.loss_acc = cal_performance(self.predict, self.target, self.criterion, self.basic_args.out_label_smoothing)

        if self.in_predict is not None and self.in_criterion is not None:
            self.loss_in_E, self.loss_in_acc = cal_performance(self.in_predict, self.target, self.in_criterion, self.basic_args.in_label_smoothing)
            loss = self.basic_args.in_loss_weight * self.loss_in_E + self.basic_args.out_loss_weight * self.loss_E
        else:
            self.loss_in_E = None
            self.loss_in_acc = None
            loss = self.loss_E

        if perform_backward:
            loss.backward()

    def zero_grad(self):
        self.optimizer_deepff.zero_grad()
        self.optimizer_transformer.zero_grad()
        self.optimizer_out.zero_grad()
        if self.in_optimizer_out is not None:
            self.in_optimizer_out.zero_grad()

    def step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.deepff.parameters(), self.basic_args.max_norm)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update deepff.')
        else:
            self.optimizer_deepff.step()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), self.basic_args.max_norm)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update transformer.')
        else:
            self.optimizer_transformer.step()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.out_prj.parameters(), self.basic_args.max_norm)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update out_prj.')
        else:
            self.optimizer_out.step()

        if self.in_out_prj is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.in_out_prj.parameters(), self.basic_args.max_norm)
            if math.isnan(grad_norm):
                print('grad norm is nan. Do not update out_prj.')
            else:
                self.in_optimizer_out.step()

    def optimize_parameters(self):

        # Perform forward
        self.forward()

        # Set zero_grad the optimizer
        self.zero_grad()

        # Perform backward
        self.backward_E()

        # Update the weight
        self.step()


class DeepFFTransformerDeepFFAM(BaseModel):
    def __init__(self, basic_args, in_deepff_args, transformer_args, out_deepff_args):
            """Initialize the DNN AM class.
            Parameters:
                basic_args: ()
                    basic_args.model_dir =

                in_deepff_args: (layer_size, hidden_act_type='relu', out_act_type = None, batch_norm = False, dropout = 0.0, name = 'DeepFFNet', version = '0.0.1')
                    deepff_args.layer_size = [d_input, 128, 128, d_model]
                    deepff_args.hidden_act_type = 'relu'
                    deepff_args.out_act_type = 'relu'
                    deepff_args.batch_norm = True
                    deepff_args.dropout = 0.1
                    deepff_args.bias = True
                    deepff_args.name = 'InDeepFFNet'
                    deepff_args.version = '0.0.1'

                transformer_args: ( d_model, d_inner, n_layers, n_head, d_k, d_v, residual_op = False, batch_norm = False, dropout = 0.0)
                    transformer_args.d_model = d_model
                    transformer_args.d_inner = d_inner
                    transformer_args.n_head = n_head
                    transformer_args.n_layers = n_layers
                    transformer_args.d_k = d_k
                    transformer_args.d_v = d_v
                    transformer_args.residual_op = True
                    transformer_args.batch_norm = True
                    transformer_args.dropout = 0.1

                out_deepff_args: (layer_size, hidden_act_type='relu', out_act_type = None, batch_norm = False, dropout = 0.0, name = 'DeepFFNet', version = '0.0.1')
                    deepff_args.layer_size = [d_input, 128, 128, d_model]
                    deepff_args.hidden_act_type = 'relu'
                    deepff_args.out_act_type = 'relu'
                    deepff_args.batch_norm = True
                    deepff_args.dropout = 0.1
                    deepff_args.bias = True
                    deepff_args.name = 'OutDeepFFNet'
                    deepff_args.version = '0.0.1'

                Loss:
                    basic_args.in_loss_type = None
                    basic_args.in_label_smoothing = 0.0
                    basic_args.in_loss_weight = 0.5
                    basic_args.in_out_act_type = None
                    basic_args.out_loss_type = 'CrossEntropyLoss'
                    basic_args.out_label_smoothing = 0.0
                    basic_args.out_loss_weight = 0.5
                    basic_args.out_out_act_type = None
                    basic_args.num_classes = 218
                    basic_args.bias = True
            """
            BaseModel.__init__(self, basic_args)
            if hasattr(basic_args, 'steps'):
                self.steps = basic_args.steps
            else:
                self.steps = 0

            self.basic_args = basic_args
            self.in_deepff_args = in_deepff_args
            self.transformer_args = transformer_args
            self.out_deepff_args = out_deepff_args

            if basic_args.in_loss_type is None:
                self.model_names = ['in_deepff', 'transformer', 'out_deepff', 'out_prj']
            else:
                self.model_names = ['in_deepff', 'transformer', 'out_deepff', 'out_prj', 'in_out_prj']

            # define and initialize the In DeepFF Net
            self.in_deepff = DeepFFNet(layer_size = in_deepff_args.layer_size, hidden_act_type = in_deepff_args.hidden_act_type, 
                                    out_act_type = in_deepff_args.out_act_type, batch_norm = in_deepff_args.batch_norm,
                                    dropout = in_deepff_args.dropout, bias = in_deepff_args.bias)
            self.in_deepff = init_net(self.in_deepff, basic_args.init_type, basic_args.init_gain, self.device)

            # define and initialize the Transformer Net
            self.transformer = nn.ModuleList([
            TransformerFFLayer(in_size = transformer_args.d_model, d_model = transformer_args.d_model, d_ff_inner = transformer_args.d_inner, query_key_size = transformer_args.d_k, 
                                value_size = transformer_args.d_v, num_head = transformer_args.n_head, layer_norm = transformer_args.batch_norm, 
                                residual_op = transformer_args.residual_op, dropout = transformer_args.dropout)
            for _ in range(transformer_args.n_layers)])
            self.transformer = init_net(self.transformer, basic_args.init_type, basic_args.init_gain, self.device)

            # define and initialize the Out DeepFF Net
            self.out_deepff = DeepFFNet(layer_size = out_deepff_args.layer_size, hidden_act_type = out_deepff_args.hidden_act_type, 
                                    out_act_type = out_deepff_args.out_act_type, batch_norm = out_deepff_args.batch_norm,
                                    dropout = out_deepff_args.dropout, bias = out_deepff_args.bias)
            self.out_deepff = init_net(self.out_deepff, basic_args.init_type, basic_args.init_gain, self.device)

            # define and initialize the out_prj Net
            if basic_args.in_loss_type is not None:
                self.in_out_prj = Outputer(in_size = in_deepff_args.layer_size[-1], out_size = basic_args.num_classes, out_act_type = basic_args.in_out_act_type, bias = basic_args.bias)
                self.in_out_prj = init_net(self.in_out_prj, basic_args.init_type, basic_args.init_gain, self.device)
            else:
                 self.in_out_prj = None

            self.out_prj = Outputer(in_size = out_deepff_args.layer_size[-1], out_size = basic_args.num_classes, out_act_type = basic_args.out_out_act_type, bias = basic_args.bias)
            self.out_prj = init_net(self.out_prj, basic_args.init_type, basic_args.init_gain, self.device)

            if basic_args.in_loss_type is None:
                self.loss_names = ['E', 'acc']
            else:
                self.loss_names = ['E', 'acc', 'in_E', 'in_acc']

            if basic_args.isTrain:

                # define loss functions of DNN AM
                if basic_args.out_loss_type.lower() == 'crossentropyloss':
                    self.criterion = torch.nn.CrossEntropyLoss(weight = basic_args.data_rate, ignore_index = IGNORE_ID).to(self.device)
                elif basic_args.out_loss_type.lower() == 'focalloss':
                    self.criterion = FocalLoss(basic_args.num_classes, self.device, ignore_index = IGNORE_ID, alpha = basic_args.data_rate).to(self.device)
                elif basic_args.out_loss_type.lower() == 'marginloss':
                    self.criterion = torch.nn.MultiMarginLoss().to(self.device)
                elif basic_args.out_loss_type.lower() == 'softmaxmarginloss':
                    self.criterion = torch.nn.MultiMarginLoss().to(self.device)
                else:
                    raise Exception('{} train_type error'.format( basic_args.out_loss_type))

                if self.in_out_prj is not None:
                    if basic_args.in_loss_type.lower() == 'crossentropyloss':
                        self.in_criterion = torch.nn.CrossEntropyLoss(weight = basic_args.data_rate, ignore_index = IGNORE_ID).to(self.device)
                    elif basic_args.in_loss_type.lower() == 'focalloss':
                        self.criterion = FocalLoss(basic_args.num_classes, self.device, ignore_index = IGNORE_ID, alpha = basic_args.data_rate).to(self.device)
                    elif basic_args.in_loss_type.lower() == 'marginloss':
                        self.in_criterion = torch.nn.MultiMarginLoss().to(self.device)
                    elif basic_args.in_loss_type.lower() == 'softmaxmarginloss':
                        self.in_criterion = torch.nn.MultiMarginLoss().to(self.device)
                    else:
                        raise Exception('{} train_type error'.format( basic_args.in_loss_type))
                else:
                    self.in_criterion = None

                # define the optimizer of In DeepFF
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_in_deepff = torch.optim.Adadelta(self.in_deepff.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_in_deepff = torch.optim.Adam(self.in_deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_in_deepff = torch.optim.SGD(self.in_deepff.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_in_deepff = torch.optim.Adam(self.in_deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_in_deepff)

                # define the optimizer of transformer
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_transformer = torch.optim.Adadelta(self.transformer.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_transformer = torch.optim.Adam(self.transformer.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_transformer = torch.optim.SGD(self.transformer.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_transformer = torch.optim.Adam(self.transformer.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_transformer)

                 # define the optimizer of In Out DeepFF
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_out_deepff = torch.optim.Adadelta(self.out_deepff.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_out_deepff = torch.optim.Adam(self.out_deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_out_deepff = torch.optim.SGD(self.out_deepff.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_out_deepff = torch.optim.Adam(self.out_deepff.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_out_deepff)

                # define the optimizer of out_prj
                if basic_args.opt_type.lower() == 'adadelta':
                    self.optimizer_out = torch.optim.Adadelta(self.out_prj.parameters(), rho=0.95, eps=1e-6)
                elif basic_args.opt_type.lower() == 'adam':
                    self.optimizer_out = torch.optim.Adam(self.out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                elif basic_args.opt_type.lower() == 'sgd':
                    self.optimizer_out = torch.optim.SGD(self.out_prj.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                else:
                    self.optimizer_out = torch.optim.Adam(self.out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                self.optimizers.append(self.optimizer_out)

                if self.in_out_prj is not None:
                    if basic_args.opt_type.lower() == 'adadelta':
                        self.in_optimizer_out = torch.optim.Adadelta(self.in_out_prj.parameters(), rho=0.95, eps=1e-6)
                    elif basic_args.opt_type.lower() == 'adam':
                        self.in_optimizer_out = torch.optim.Adam(self.in_out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                    elif basic_args.opt_type.lower() == 'sgd':
                        self.in_optimizer_out = torch.optim.SGD(self.in_out_prj.parameters(), lr=basic_args.lr, momentum=0.9, weight_decay=1e-4)
                    else:
                        self.in_optimizer_out = torch.optim.Adam(self.in_out_prj.parameters(), lr=basic_args.lr, betas=(basic_args.beta1, 0.999))
                    self.optimizers.append(self.in_optimizer_out)
                else:
                    self.in_optimizer_out = None
            self.setup(basic_args)

    @classmethod
    def load_model(cls, model_path, continue_from_name = 'best', given_model_configure = None, gpu_ids = [1], isTrain = False):

        if continue_from_name is None:
            exit("ERROR: continue_from_model is None")

        load_filename = 'model_%s.configure' % (continue_from_name)
        configure_path = os.path.join(model_path, load_filename)

        if not os.path.exists(configure_path):
            exit("ERROR: %s is not existed" % (configure_path))

        model_configure = torch.load(configure_path, map_location=lambda storage, loc: storage)
        basic_args = model_configure['basic_args']
        basic_args.model_dir = model_path
        basic_args.continue_from_name = continue_from_name
        basic_args.gpu_ids = gpu_ids
        basic_args.isTrain = isTrain

        if given_model_configure is not None:

            if hasattr(given_model_configure, 'data_rate'):
                basic_args.data_rate = given_model_configure.data_rate

            if hasattr(given_model_configure, 'save_freq_steps'):
                basic_args.save_freq_steps = given_model_configure.save_freq_steps
            if hasattr(given_model_configure, 'save_by_steps'):
                basic_args.save_by_steps = given_model_configure.save_by_steps
            if hasattr(given_model_configure, 'print_freq_steps'):
                basic_args.print_freq_steps = given_model_configure.print_freq_steps
            if hasattr(given_model_configure, 'validate_freq_steps'):
                basic_args.validate_freq_steps = given_model_configure.validate_freq_steps

            if hasattr(given_model_configure, 'epochs'):
                basic_args.epochs = given_model_configure.epochs
            if hasattr(given_model_configure, 'opt_type'):
                basic_args.opt_type = given_model_configure.opt_type
            if hasattr(given_model_configure, 'lr_policy'):
                basic_args.lr_policy = given_model_configure.lr_policy
            if hasattr(given_model_configure, 'lr'):
                basic_args.lr = given_model_configure.lr
            if hasattr(given_model_configure, 'beta1'):
                basic_args.beta1 = given_model_configure.beta1
            if hasattr(given_model_configure, 'max_norm'):
                basic_args.max_norm = given_model_configure.max_norm

            if hasattr(given_model_configure, 'lr_freq_steps'):
                basic_args.lr_freq_steps = given_model_configure.lr_freq_steps

            if hasattr(given_model_configure, 'lr_factor_freq_step'):
                basic_args.lr_factor_freq_step = given_model_configure.lr_factor_freq_step
            if hasattr(given_model_configure, 'lr_factor'):
                basic_args.lr_factor = given_model_configure.lr_factor
            if hasattr(given_model_configure, 'warmup_step'):
                basic_args.warmup_step = given_model_configure.warmup_step

            if hasattr(given_model_configure, 'niter'):
                basic_args.niter = given_model_configure.niter
            if hasattr(given_model_configure, 'niter_decay'):
                basic_args.niter_decay = given_model_configure.niter_decay

            if hasattr(given_model_configure, 'lr_decay_iters'):
                basic_args.lr_decay_iters = given_model_configure.lr_decay_iters

            if hasattr(given_model_configure, 'lr_reduce_factor'):
                basic_args.lr_reduce_factor = given_model_configure.lr_reduce_factor
            if hasattr(given_model_configure, 'lr_reduce_threshold'):
                basic_args.lr_reduce_threshold = given_model_configure.lr_reduce_threshold
            if hasattr(given_model_configure, 'step_patience'):
                basic_args.step_patience = given_model_configure.step_patience
            if hasattr(given_model_configure, 'min_lr'):
                basic_args.min_lr = given_model_configure.min_lr

            if hasattr(given_model_configure, 'visdom_lr'):
                basic_args.visdom_lr = given_model_configure.visdom_lr
            if hasattr(given_model_configure, 'visdom'):
                basic_args.visdom = given_model_configure.visdom
            if hasattr(given_model_configure, 'visdom_id'):
                basic_args.visdom_id = given_model_configure.visdom_id
            if hasattr(given_model_configure, 'display_server'):
                basic_args.display_server = given_model_configure.display_server
            if hasattr(given_model_configure, 'display_port'):
                basic_args.display_port = given_model_configure.display_port
            if hasattr(given_model_configure, 'visdom_freq_steps'):
                basic_args.visdom_freq_steps = given_model_configure.visdom_freq_steps

            if hasattr(given_model_configure, 'verbose'):
                basic_args.verbose = given_model_configure.verbose

        in_deepff_args = model_configure['in_deepff_args']
        transformer_args = model_configure['transformer_args']
        out_deepff_args = model_configure['out_deepff_args']

        basic_args.steps = model_configure['tr_steps']
        model = cls(basic_args, in_deepff_args, transformer_args, out_deepff_args)
        model.load_networks(continue_from_name)
        model.steps = model_configure['tr_steps']

        model_state = {
            'epoch': model_configure['epoch'],
            'tr_steps': model_configure['tr_steps'],
            'val_steps': model_configure['val_steps'],
            'tr_loss': model_configure['tr_loss'],
            'tr_acc': model_configure['tr_acc'],
            'val_loss': model_configure['val_loss'],
            'val_acc': model_configure['val_acc']
        }

        return model, model_state, basic_args, in_deepff_args, transformer_args, out_deepff_args

    def save_model(self, suffix_name, epoch, val_steps, tr_loss = None, tr_acc = None, val_loss = None, val_acc = None):

        configure_package = {
            'epoch': epoch,
            'tr_steps': self.steps,
            'val_steps': val_steps,
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'basic_args': self.basic_args,
            'in_deepff_args': self.in_deepff_args,
            'transformer_args': self.transformer_args,
            'out_deepff_args': self.out_deepff_args
        }
        save_filename = 'model_%s.configure' % (suffix_name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(configure_package, save_path)
        self.save_networks(suffix_name)

    def set_input(self, padded_input, target = None, input_lengths = None):
        """
        Args:
            padded_input: (num_block, num_frame, d_x)
            target: (num_block, num_frame)
            input_lengths: (num_block, 1)
        Returns:
            enc_output: (num_block, num_frame, d_y)
        """
        self.steps = self.steps + 1

        self.padded_input = padded_input.to(self.device)

        if target is not None:
            self.target = target.to(self.device)
        else:
            self.target = None

        if input_lengths is not None:
            self.input_lengths = input_lengths
            self.non_pad_mask = get_non_pad_mask(self.padded_input, input_lengths = self.input_lengths)
            length = padded_input.size(1)
            self.slf_attn_mask = get_attn_pad_mask(self.padded_input, self.input_lengths, length)
        else:
            self.input_lengths = None
            self.non_pad_mask = None

    def inference(self, padded_input, input_lengths = None):

        self.padded_input = padded_input.to(self.device)

        if input_lengths is not None:
            self.input_lengths = input_lengths
            self.non_pad_mask = get_non_pad_mask(self.padded_input, input_lengths = self.input_lengths)
            length = padded_input.size(1)
            self.slf_attn_mask = get_attn_pad_mask(self.padded_input, self.input_lengths, length)
        else:
            self.input_lengths = None
            self.non_pad_mask = None
            self.slf_attn_mask = None

        # enc_output: (num_block, num_frame, num_phone)
        enc_output = self.in_deepff(self.padded_input)
        if self.in_out_prj is not None:
            in_predict = self.in_out_prj(enc_output)
            in_predict_id = in_predict.max(2, keepdim=True)[1]
        else:
            in_predict = None
            in_predict_id = None

        for transformer_layer in self.transformer:
            enc_output, _ = transformer_layer(enc_output, non_pad_mask = self.non_pad_mask, slf_attn_mask = self.slf_attn_mask)

        enc_output = self.out_deepff(enc_output)

        predict = self.out_prj(enc_output)

        predict_id = predict.max(2, keepdim=True)[1]

        return predict, predict_id, in_predict, in_predict_id

    def forward(self):

        # enc_output: (num_block, num_frame, num_phone)
        enc_output = self.in_deepff(self.padded_input)
        if self.in_out_prj is not None:
            self.in_predict = self.in_out_prj(enc_output)
        else:
            self.in_predict = None

        for transformer_layer in self.transformer:
            enc_output, _ = transformer_layer(enc_output, non_pad_mask = self.non_pad_mask, slf_attn_mask = self.slf_attn_mask)

        enc_output = self.out_deepff(enc_output)

        self.predict = self.out_prj(enc_output)

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.backward_E(perform_backward = False)

    def backward_E(self, perform_backward = True):

        self.loss_E, self.loss_acc = cal_performance(self.predict, self.target, self.criterion, self.basic_args.out_label_smoothing)

        if self.in_predict is not None and self.in_criterion is not None:
            self.loss_in_E, self.loss_in_acc = cal_performance(self.in_predict, self.target, self.in_criterion, self.basic_args.in_label_smoothing)
            loss = self.basic_args.in_loss_weight * self.loss_in_E + self.basic_args.out_loss_weight * self.loss_E
        else:
            self.loss_in_E = None
            self.loss_in_acc = None
            loss = self.loss_E

        if perform_backward:
            loss.backward()

    def zero_grad(self):
        self.optimizer_in_deepff.zero_grad()
        self.optimizer_transformer.zero_grad()
        self.optimizer_out_deepff.zero_grad()
        self.optimizer_out.zero_grad()
        if self.in_optimizer_out is not None:
            self.in_optimizer_out.zero_grad()

    def step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.in_deepff.parameters(), self.basic_args.max_norm)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update in_deepff.')
        else:
            self.optimizer_in_deepff.step()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), self.basic_args.max_norm)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update transformer.')
        else:
            self.optimizer_transformer.step()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.out_deepff.parameters(), self.basic_args.max_norm)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update out_deepff.')
        else:
            self.optimizer_out_deepff.step()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.out_prj.parameters(), self.basic_args.max_norm)
        if math.isnan(grad_norm):
            print('grad norm is nan. Do not update out_prj.')
        else:
            self.optimizer_out.step()

        if self.in_out_prj is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.in_out_prj.parameters(), self.basic_args.max_norm)
            if math.isnan(grad_norm):
                print('grad norm is nan. Do not update out_prj.')
            else:
                self.in_optimizer_out.step()

    def optimize_parameters(self):

        # Perform forward
        self.forward()

        # set zero_grad to the optimizer
        self.zero_grad()

        # Perform backward
        self.backward_E()

        # Update the weight
        self.step()


class TeacherStudent(object):
    def __init__(self, ts_args, teacher, student):
        """Initialize the DNN AM class.
        Parameters:
            basic_args: ()
                basic_args.model_dir =

            teacher: a large model to guid the student to learn

            student: a small model to learn the teacher model
        """
        if hasattr(ts_args, 'steps'):
            self.steps = ts_args.steps
        else:
            self.steps = 0

        self.ts_args = ts_args

        self.teacher = teacher
        self.student = student

        self.teacher.save_dir = ts_args.model_dir
        self.student.save_dir = ts_args.model_dir
        self.device = self.teacher.device

        # define loss functions of student-teacher model
        if ts_args.ts_loss_type.lower() == 'soft_crossentropyloss':
            self.criterion = None
        elif ts_args.ts_loss_type.lower() == 'kldivloss':
            self.criterion = torch.nn.KLDivLoss().to(self.device)
        else:
            raise Exception('{} train_type error'.format( ts_args.ts_loss_type))

        self.loss_names = ['E', 'acc', 'tsE']

    def set_input(self, padded_input, target = None, input_lengths = None):
        """
        Args:
            padded_input: (num_block, num_frame, d_x)
            target: (num_block, num_frame)
            input_lengths: (num_block, 1)
        Returns:
            enc_output: (num_block, num_frame, d_y)
        """
        self.steps = self.steps + 1
        self.teacher.set_input(padded_input, target, input_lengths)
        self.student.set_input(padded_input, target, input_lengths)

    def eval(self):
        self.teacher.eval()
        self.student.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.backward_E(perform_backward = False)

    def get_current_losses(self):
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def train(self):
        self.teacher.train()
        self.student.train()

    def update_learning_rate(self):
        self.student.update_learning_rate()

    def set_lr_factor(self, lr_factor):
        self.student.set_lr_factor(lr_factor)

    def forward(self):
        # enc_output: (num_block, num_frame, num_phone)
        #with torch.no_grad():
        self.teacher.forward()
        self.student.forward()

    def backward_E(self, perform_backward = True):

        self.loss_E, self.loss_acc = cal_performance(self.student.predict, self.student.target, self.student.criterion, self.student.basic_args.out_label_smoothing)

        if len( self.teacher.predict.size() ) > 2:
            if not self.teacher.predict.is_contiguous():
                teacher_predict = self.teacher.predict.contiguous().view( -1, self.teacher.predict.size(-1) )
            else:
                teacher_predict = self.teacher.predict.view( -1, self.teacher.predict.size(-1) )

        if len( self.student.predict.size() ) > 2:
            if not self.student.predict.is_contiguous():
                student_predict = self.student.predict.contiguous().view( -1, self.student.predict.size(-1) )
            else:
                student_predict = self.student.predict.view( -1, self.student.predict.size(-1) )

        num_frame = teacher_predict.size(0)


        teacher_predict = F.softmax(teacher_predict, dim = -1) # target
        student_predict = F.log_softmax(student_predict, dim = -1) # predict

        if self.ts_args.ts_loss_type.lower() == 'soft_crossentropyloss':
            self.loss_tsE = -( teacher_predict * student_predict ).sum(dim = -1)
            #self.loss_tsE = self.loss_tsE / num_frame
            self.loss_tsE = self.loss_tsE.mean()
        elif self.ts_args.ts_loss_type.lower() == 'KLDivLoss':
            self.loss_tsE = self.criterion(student_predict, teacher_predict)
        else:
            self.loss_tsE = self.criterion(student_predict, teacher_predict)

        loss = self.ts_args.ts_loss_weight * self.loss_tsE + (1.0 - self.ts_args.ts_loss_weight) * self.loss_E

        if perform_backward:
            loss.backward()

    def optimize_parameters(self):

        # Perform forward on teacher and student
        self.forward()

        # Set zero_grad to the student parameters
        self.student.zero_grad()

        # Perform backward
        self.backward_E()

        # Update the weigth of student
        self.student.step()

    def save_model(self, suffix_name, epoch, val_steps, tr_loss = None, tr_acc = None, val_loss = None, val_acc = None):

        configure_package = {
            'epoch': epoch,
            'tr_steps': self.steps,
            'val_steps': val_steps,
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_dir': self.ts_args.model_dir
        }
        save_filename = 'ts_model_%s.configure' % (suffix_name)
        save_path = os.path.join(self.ts_args.model_dir, save_filename)
        torch.save(configure_package, save_path)

        given_suffix_name = suffix_name
        suffix_name = given_suffix_name + "_student"
        self.student.save_model(suffix_name, epoch, val_steps, tr_loss, tr_acc, val_loss, val_acc)

        suffix_name = given_suffix_name + "_teacher"
        self.teacher.save_model(suffix_name, epoch, val_steps, tr_loss, tr_acc, val_loss, val_acc)

    @classmethod
    def load_model(cls, ts_args, gpu_ids = [1], isTrain = False):

        teacher_model_path          = ts_args.teacher_model_path
        student_model_path          = ts_args.student_model_path
        teacher_continue_from_name  = ts_args.teacher_continue_from_name
        student_continue_from_name  = ts_args.student_continue_from_name

        if teacher_continue_from_name is None:
            exit("ERROR: teacher_continue_from_model is None")
        if student_continue_from_name is None:
            exit("ERROR: student_continue_from_model is None")

        load_filename = 'ts_model_%s.configure' % (ts_args.continue_from_name)
        configure_path = os.path.join(ts_args.model_dir, load_filename)
        if os.path.exists(configure_path):
            model_configure = torch.load(configure_path, map_location=lambda storage, loc: storage)
            ts_args.steps = model_configure['tr_steps']
        else:
            model_configure = None
            ts_args.steps = 0

        ## load the teacher model
        if ts_args.teacher_model.lower() == 'deepffam':
            teacher_model, _, basic_args, _ = DeepFFAM.load_model(teacher_model_path, teacher_continue_from_name, ts_args, gpu_ids, isTrain)
        elif ts_args.teacher_model.lower() == 'deepfftransformeram':
            teacher_model, _, basic_args, _, _ = DeepFFTransformerAM.load_model(teacher_model_path, teacher_continue_from_name, ts_args, gpu_ids, isTrain)
        else:
            teacher_model, _, basic_args, _, _, _ = DeepFFTransformerDeepFFAM.load_model(teacher_model_path, teacher_continue_from_name, ts_args, gpu_ids, isTrain)
        ts_args.d_model = basic_args.d_model
        teacher_model.setup(ts_args)

        ## load the student model
        if ts_args.student_model.lower() == 'deepffam':
            student_model, _, basic_args, _ = DeepFFAM.load_model(student_model_path, student_continue_from_name, ts_args, gpu_ids, isTrain)
        elif ts_args.teacher_model.lower() == 'deepfftransformeram':
            student_model, _, basic_args, _, _ = DeepFFTransformerAM.load_model(student_model_path, student_continue_from_name, ts_args, gpu_ids, isTrain)
        else:
            student_model, _, basic_args, _, _, _ = DeepFFTransformerDeepFFAM.load_model(student_model_path, student_continue_from_name, ts_args, gpu_ids, isTrain)
        ts_args.d_model = basic_args.d_model
        student_model.setup(ts_args)

        ## build the teacher-student model
        ts_model = cls( ts_args, teacher_model, student_model )

        return ts_model, model_configure
