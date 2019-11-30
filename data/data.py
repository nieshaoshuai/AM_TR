import os
import numpy as np
import torch

from model.loss import IGNORE_ID
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from audio.audioparser import FeatLabelParser
import random
import progressbar
import gc
import matplotlib.pyplot as plt

batch_conf_example = dict(exp_path='./',
                              delta_order=0,
                              left_context_width=0,
                              right_context_width=0,
                              normalize_type=1,
                              max_num_utt_cmvn=100000)

class BaseDataset(Dataset, FeatLabelParser):
    def __init__(self, opt, speech_dir, noisy_dir, noise_dir, dataset = 'train', cmvn_file = None, shuffle = True):
        super(BaseDataset, self).__init__()
        
        self.opt = opt
        self.exp_path = opt.exp_path
        if not os.path.isdir(self.exp_path):
            raise Exception(self.exp_path + ' isn.t a path!')
        self.speech_dir = os.path.join(speech_dir, dataset)
        self.noisy_dir = os.path.join(noisy_dir, dataset)
        self.noise_dir = os.path.join(noise_dir, dataset)
        
        # basic parameters
        self.shuffle             = shuffle
        self.delta_order         = opt.delta_order
        self.left_context_width  = opt.left_context_width
        self.right_context_width = opt.right_context_width

        self.normalize_type = opt.normalize_type
        self.num_utt_cmvn   = opt.num_utt_cmvn
        if cmvn_file is None:
            self.cmvn_file  = os.path.join(self.exp_path, 'cmvn.npy')
        else:
            self.cmvn_file  = cmvn_file
        self.cmvn = None

        # dataset ratio configure
        self.speech_rate = self.opt.speech_rate
        self.noisy_rate  = self.opt.noisy_rate
        self.noise_rate  = self.opt.noise_rate
        self.babble_rate = self.opt.babble_rate
        self.music_rate  = self.opt.music_rate

        self.ignore_label_id = IGNORE_ID
        self.noise_label_id  = 0
        self.babble_label_id = 0
        self.music_label_id  = 0
        
        # construct feats and labels for clean speech
        feats_file = os.path.join(self.speech_dir, 'feats.scp')
        self.speech_feats = []
        with open(feats_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    print('Error %s' % (line))
                    continue
                utt_key = splits[0]
                feat_path = splits[1]
                self.speech_feats.append((utt_key, feat_path))
                
        labels_file = os.path.join(self.speech_dir, 'labels.scp')        
        self.speech_labels = []         
        with open(labels_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    print('Error %s' % (line))
                    continue
                utt_key = splits[0]
                label_path = splits[1]
                self.speech_labels.append((utt_key, label_path))
        self.speech_size = len(self.speech_feats)
        
        # construct feats and labels for noisy
        feats_file = os.path.join(self.noisy_dir, 'feats.scp')
        self.noisy_feats = []
        with open(feats_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    print('Error %s' % (line))
                    continue
                utt_key = splits[0]
                feat_path = splits[1]
                self.noisy_feats.append((utt_key, feat_path))
                
        labels_file = os.path.join(self.noisy_dir, 'labels.scp')        
        self.noisy_labels = []         
        with open(labels_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    print('Error %s' % (line))
                    continue
                utt_key = splits[0]
                label_path = splits[1]
                self.noisy_labels.append((utt_key, label_path))
        self.noisy_size = len(self.noisy_feats)
        
        # construct feats and labels for noise
        feats_file = os.path.join(self.noise_dir, 'noise_feats.scp')
        self.noise_feats = []
        self.noise_labels = []  
        with open(feats_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    print('Error %s' % (line))
                    continue
                utt_key = splits[0]
                feat_path = splits[1]
                self.noise_feats.append((utt_key, feat_path))
                self.noise_labels.append((utt_key, self.noise_label_id))
        self.noise_size = len(self.noise_feats)

        # construct feats and labels for babble noise
        feats_file = os.path.join(self.noise_dir, 'babble_feats.scp')
        self.babble_feats = []
        self.babble_labels = []
        with open(feats_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    print('Error %s' % (line))
                    continue
                utt_key = splits[0]
                feat_path = splits[1]
                self.babble_feats.append((utt_key, feat_path))
                self.babble_labels.append((utt_key, self.babble_label_id))
        self.babble_size = len(self.babble_feats)

        # construct feats and labels for noise
        feats_file = os.path.join(self.noise_dir, 'music_feats.scp')
        self.music_feats = []
        self.music_labels = []
        with open(feats_file) as f:
            for line in f.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    print('Error %s' % (line))
                    continue
                utt_key = splits[0]
                feat_path = splits[1]
                self.music_feats.append((utt_key, feat_path))
                self.music_labels.append((utt_key, self.music_label_id))
        self.music_size = len(self.music_feats)

        # construct reading idx
        self.num_utts       = int(self.speech_rate > 0) * self.speech_size + int(self.noisy_rate > 0) * self.noisy_size + \
                              int(self.noise_rate > 0) * self.noise_size + int(self.babble_rate > 0) * self.babble_size + int(self.music_rate > 0) * self.music_size
        self.one_time_utts  = self.speech_rate + self.noisy_rate + self.noise_rate + self.babble_rate + self.music_rate
        print("speech_size = %d, noisy_size = %d noise_size = %d babble_size = %d, music_size = %d" % (self.speech_size, self.noisy_size, self.noise_size, self.babble_size, self.music_size))
        print("%s set contain %d utts, one turn get %d utts" % (dataset, self.num_utts, self.one_time_utts))
        
        if shuffle:
            self.rand_speech_idx = np.random.permutation(self.speech_size)
            self.rand_noisy_idx  = np.random.permutation(self.noisy_size)
            self.rand_noise_idx  = np.random.permutation(self.noise_size)
            self.rand_babble_idx = np.random.permutation(self.babble_size)
            self.rand_music_idx  = np.random.permutation(self.music_size)
        else:
            self.rand_speech_idx = range(self.speech_size)
            self.rand_noisy_idx  = range(self.noisy_size)
            self.rand_noise_idx  = range(self.noise_size)
            self.rand_babble_idx = range(self.babble_size)
            self.rand_music_idx  = range(self.music_size)

        for utt_key, feat_path in self.speech_feats:
            in_feat = self.load_feat(feat_path, self.delta_order)
            if in_feat is not None:
                break
        self.feat_size = in_feat.shape[1]
        self.in_size = self.feat_size * (self.left_context_width + self.right_context_width + 1)

        if self.normalize_type == 1:
            if opt.cmvn_file is None:
                self.cmvn_file = os.path.join(self.exp_path, 'cmvn.npy')
            else:
                self.cmvn_file = opt.cmvn_file
            self.cmvn = self.loading_cmvn(self.cmvn_file)

    def rand_segment(self, feat, segment_length):
        if feat is None:
            return None
        num_frame = feat.shape[0]
        if segment_length < num_frame:
            index = random.randint(0, (num_frame - segment_length) - 1)
            segment_feat = feat[index:index + segment_length, :]
        else:
            segment_feat = None
        return segment_feat

    def compute_cmvn(self):
        cmvn_num = min(self.num_utts, self.num_utt_cmvn)
        print(">> compute cmvn using {0} utterance ".format(cmvn_num))

        sum_all         = np.zeros(shape=[1, self.feat_size], dtype = np.float64)
        sum_square_all  = np.zeros(shape=[1, self.feat_size], dtype = np.float64)
        cmvn            = np.zeros(shape=[2, self.feat_size], dtype = np.float32)

        have_scan_utt   = 0
        frame_count     = 0
        p               = progressbar.ProgressBar(cmvn_num)
        p.start()
        while have_scan_utt < cmvn_num:
            for i in range(self.speech_rate):
                utt_idx = random.randint(0, self.speech_size - 1)
                _, feat_path = self.speech_feats[utt_idx]

                in_feat = self.load_feat(feat_path, self.delta_order)
                if in_feat is None:
                    continue

                frame_count += np.shape(in_feat)[0]
                sum_1utt = np.sum(in_feat, axis=0)
                sum_all = np.add(sum_all, sum_1utt)

                feature_mat_square = np.square(in_feat)
                sum_square_1utt = np.sum(feature_mat_square, axis=0)
                sum_square_all = np.add(sum_square_all, sum_square_1utt)

            for i in range(self.noisy_rate):
                utt_idx = random.randint(0, self.noisy_size - 1)
                _, feat_path = self.noisy_feats[utt_idx]

                in_feat = self.load_feat(feat_path, self.delta_order)
                if in_feat is None:
                    continue

                frame_count += np.shape(in_feat)[0]
                sum_1utt = np.sum(in_feat, axis=0)
                sum_all = np.add(sum_all, sum_1utt)

                feature_mat_square = np.square(in_feat)
                sum_square_1utt = np.sum(feature_mat_square, axis=0)
                sum_square_all = np.add(sum_square_all, sum_square_1utt)

            for i in range(self.noise_rate):
                utt_idx = random.randint(0, self.noise_size - 1)
                _, feat_path = self.noise_feats[utt_idx]

                in_feat = self.load_feat(feat_path, self.delta_order)
                if in_feat is None:
                    continue

                frame_count += np.shape(in_feat)[0]
                sum_1utt = np.sum(in_feat, axis=0)
                sum_all = np.add(sum_all, sum_1utt)

                feature_mat_square = np.square(in_feat)
                sum_square_1utt = np.sum(feature_mat_square, axis=0)
                sum_square_all = np.add(sum_square_all, sum_square_1utt)

            for i in range(self.babble_rate):
                utt_idx = random.randint(0, self.babble_size - 1)
                _, feat_path = self.babble_feats[utt_idx]

                in_feat = self.load_feat(feat_path, self.delta_order)
                if in_feat is None:
                    continue

                frame_count += np.shape(in_feat)[0]
                sum_1utt = np.sum(in_feat, axis=0)
                sum_all = np.add(sum_all, sum_1utt)

                feature_mat_square = np.square(in_feat)
                sum_square_1utt = np.sum(feature_mat_square, axis=0)
                sum_square_all = np.add(sum_square_all, sum_square_1utt)

            for i in range(self.music_rate):
                utt_idx = random.randint(0, self.music_size - 1)
                _, feat_path = self.music_feats[utt_idx]

                in_feat = self.load_feat(feat_path, self.delta_order)
                if in_feat is None:
                    continue

                frame_count += np.shape(in_feat)[0]
                sum_1utt = np.sum(in_feat, axis=0)
                sum_all = np.add(sum_all, sum_1utt)

                feature_mat_square = np.square(in_feat)
                sum_square_1utt = np.sum(feature_mat_square, axis=0)
                sum_square_all = np.add(sum_square_all, sum_square_1utt)

            have_scan_utt = have_scan_utt + self.one_time_utts
            p.update(min(have_scan_utt, cmvn_num))
        p.finish()
        
        print(">> compute cmvn using {0} samples ".format(frame_count))
        mean = sum_all / frame_count
        var = sum_square_all / frame_count - np.square(mean)
        cmvn[0, :] = -mean
        cmvn[1, :] = 1.0 / np.real(np.sqrt(var))
        return cmvn

    def loading_cmvn(self, cmvn_file):
        if os.path.exists(cmvn_file):
            cmvn = np.load(cmvn_file)
            if cmvn.shape[1] == self.feat_size:
                print ('load cmvn from {}'.format(cmvn_file))
            else:
                cmvn = self.compute_cmvn()
                np.save(cmvn_file, cmvn)
                print ('original cmvn is wrong, so save new cmvn to {}'.format(cmvn_file))
        else:
            cmvn = self.compute_cmvn()
            np.save(cmvn_file, cmvn)
            print ('save cmvn to {}'.format(cmvn_file))
        return cmvn

class BlockDataset(BaseDataset):
    def __init__(self, data_args, speech_dir, noisy_dir, noise_dir, dataset = 'train', cmvn_file = None, shuffle = True):
        '''
        Prepare data for training
        
        return block_data, block_label
            block_data: (num_block, block_length, in_size)
            block_label: (num_block, block_length)
            block_length: (num_block, )
        '''
        super(BlockDataset, self).__init__(data_args, speech_dir, noisy_dir, noise_dir, dataset, cmvn_file, shuffle)
        
        self.data_args      = data_args
        self.block_length   = max(1, data_args.block_length)
        self.block_shift    = max(1, data_args.block_shift)
        self.block_width    = self.in_size
    
    def data_parser(self, feat_path, label):
        
        feat = self.parse_feat(feat_path, delta_order = self.delta_order, cmvn = self.cmvn, left_context_width = self.left_context_width, right_context_width = self.right_context_width)
        num_frame = feat.shape[0]
        
        if isinstance(label, str):
            label = self.parse_label(label, num_frame)
        else:
            label = self.parse_label(None, num_frame, label)
        
        return feat, label, num_frame

    def __getitem__(self, index):
        # return
        #   block_feats:  (num_block, block_length, block_width)
        #   block_labels: (num_block, block_length)
        #   block_length: (num_block, 1)

        block_feats  = np.zeros(shape=(0, self.block_length, self.block_width), dtype = np.float32)
        block_labels = np.zeros(shape=(0, self.block_length), dtype = np.int32)
        block_length = []

        for i in range(self.speech_rate):
            #utt_idx = random.randint(0, self.speech_size - 1)
            utt_idx = self.rand_speech_idx[(index * self.speech_rate + i) % self.speech_size]
            
            feat_key, feat_path    = self.speech_feats[utt_idx]
            label_key, label_path  = self.speech_labels[utt_idx]
            
            if feat_key.strip() != label_key.strip():
                print("feat key does not match label key")
                continue
            
            feats, labels, num_frame = self.data_parser(feat_path, label_path)
            for start in range(0, num_frame, self.block_shift):
                end = start + self.block_length
                if end < num_frame:
                    sub_feats   = feats[start:end, :][np.newaxis, :, :]
                    sub_labels  = labels[0, start:end][np.newaxis, :]
                    frames      = end - start
                else:
                    sub_feats = np.zeros(shape = (1, self.block_length, self.block_width), dtype = np.float32)
                    sub_labels = np.zeros(shape = (1, self.block_length), dtype = np.int32) + self.ignore_label_id
                    end = num_frame
                    start = max(0, end - self.block_length)

                    frames = end - start
                    sub_feats[0, :frames, :] = feats[start:, :][np.newaxis, :, :]
                    sub_labels[0, :frames] = labels[0, start:]

                block_feats = np.concatenate((block_feats, sub_feats), axis=0)
                block_labels = np.concatenate((block_labels, sub_labels), axis=0)
                block_length.append(frames)

        for i in range(self.noisy_rate):
            #utt_idx = random.randint(0, self.noisy_size - 1)
            utt_idx = self.rand_noisy_idx[(index * self.noisy_rate + i) % self.noisy_size]

            feat_key, feat_path    = self.noisy_feats[utt_idx]
            label_key, label_path  = self.noisy_labels[utt_idx]
            
            if feat_key.strip() != label_key.strip():
                print("feat key does not match label key")
                continue
                
            feats, labels, num_frame = self.data_parser(feat_path, label_path)
            for start in range(0, num_frame, self.block_shift):
                end = start + self.block_length
                if end < num_frame:
                    sub_feats = feats[start:end, :][np.newaxis, :, :]
                    sub_labels = labels[0, start:end][np.newaxis, :]
                    frames = end - start
                else:
                    sub_feats = np.zeros(shape = (1, self.block_length, self.block_width), dtype=np.float32)
                    sub_labels = np.zeros(shape = (1, self.block_length), dtype=np.int32) + self.ignore_label_id
                    end = num_frame
                    start = max(0, end - self.block_length)

                    frames = end - start
                    sub_feats[0, :frames, :] = feats[start:, :][np.newaxis, :, :]
                    sub_labels[0, :frames] = labels[0, start:]

                block_feats = np.concatenate((block_feats, sub_feats), axis=0)
                block_labels = np.concatenate((block_labels, sub_labels), axis=0)
                block_length.append(frames)

        for i in range(self.noise_rate):
            #utt_idx = random.randint(0, self.noise_size - 1)
            utt_idx = self.rand_noise_idx[(index * self.noise_rate + i) % self.noise_size]

            feat_key, feat_path    = self.noise_feats[utt_idx]
            label_key, label_path  = self.noise_labels[utt_idx]
            
            if feat_key.strip() != label_key.strip():
                print("feat key does not match label key")
                continue
                
            feats, labels, num_frame = self.data_parser(feat_path, label_path)
            
            for start in range(0, num_frame, self.block_shift):
                end = start + self.block_length
                if end < num_frame:
                    sub_feats = feats[start:end, :][np.newaxis, :, :]
                    sub_labels = labels[0, start:end][np.newaxis, :]
                    frames = end - start
                else:
                    sub_feats = np.zeros(shape = (1, self.block_length, self.block_width), dtype=np.float32)
                    sub_labels = np.zeros(shape = (1, self.block_length), dtype=np.int32) + self.ignore_label_id
                    end = num_frame
                    start = max(0, end - self.block_length)

                    frames = end - start
                    sub_feats[0, :frames, :] = feats[start:, :][np.newaxis, :, :]
                    sub_labels[0, :frames] = labels[0, start:]

                block_feats = np.concatenate((block_feats, sub_feats), axis=0)
                block_labels = np.concatenate((block_labels, sub_labels), axis=0)
                block_length.append(frames)
            
        for i in range(self.babble_rate):
            #utt_idx = random.randint(0, self.babble_size - 1)
            utt_idx = self.rand_babble_idx[(index * self.babble_rate + i) % self.babble_size]

            feat_key, feat_path    = self.babble_feats[utt_idx]
            label_key, label_path  = self.babble_labels[utt_idx]
            
            if feat_key.strip() != label_key.strip():
                print("feat key does not match label key")
                continue
                
            feats, labels, num_frame = self.data_parser(feat_path, label_path)
            for start in range(0, num_frame, self.block_shift):
                end = start + self.block_length
                if end < num_frame:
                    sub_feats = feats[start:end, :][np.newaxis, :, :]
                    sub_labels = labels[0, start:end][np.newaxis, :]
                    frames = end - start
                else:
                    sub_feats = np.zeros(shape = (1, self.block_length, self.block_width), dtype=np.float32)
                    sub_labels = np.zeros(shape = (1, self.block_length), dtype=np.int32) + self.ignore_label_id
                    end = num_frame
                    start = max(0, end - self.block_length)

                    frames = end - start
                    sub_feats[0, :frames, :] = feats[start:, :][np.newaxis, :, :]
                    sub_labels[0, :frames] = labels[0, start:]

                block_feats = np.concatenate((block_feats, sub_feats), axis=0)
                block_labels = np.concatenate((block_labels, sub_labels), axis=0)
                block_length.append(frames)

        for i in range(self.music_rate):
            #utt_idx = random.randint(0, self.music_size - 1)
            utt_idx = self.rand_music_idx[(index * self.music_rate + i) % self.music_size]

            feat_key, feat_path    = self.music_feats[utt_idx]
            label_key, label_path  = self.music_labels[utt_idx]
            
            if feat_key.strip() != label_key.strip():
                print("feat key does not match label key")
                continue
                
            feats, labels, num_frame = self.data_parser(feat_path, label_path)
            for start in range(0, num_frame, self.block_shift):
                end = start + self.block_length
                if end < num_frame:
                    sub_feats = feats[start:end, :][np.newaxis, :, :]
                    sub_labels = labels[0, start:end][np.newaxis, :]
                    frames = end - start
                else:
                    sub_feats = np.zeros(shape = (1, self.block_length, self.block_width), dtype=np.float32)
                    sub_labels = np.zeros(shape = (1, self.block_length), dtype=np.int32) + self.ignore_label_id
                    end = num_frame
                    start = max(0, end - self.block_length)

                    frames = end - start
                    sub_feats[0, :frames, :] = feats[start:, :][np.newaxis, :, :]
                    sub_labels[0, :frames] = labels[0, start:]

                block_feats = np.concatenate((block_feats, sub_feats), axis=0)
                block_labels = np.concatenate((block_labels, sub_labels), axis=0)
                block_length.append(frames)
            
        block_feats     = torch.FloatTensor(block_feats)
        block_labels    = torch.LongTensor(block_labels)
        block_length    = torch.LongTensor(np.array(block_length))
        
        return block_feats, block_labels, block_length
       
    def __len__(self):
        #return self.speech_size
        return int(self.num_utts / self.one_time_utts)

def _block_collate_fn(batch):
    def func(p):
        if p[0] is None:
            return 0
        return p[0].size(1)

    # input: 
    #   block_feats:  (num_block, block_length, block_width)
    #   block_labels: (num_block, block_length)
    #   block_length: (num_block, 1)

    num_block = 0
    for sample in batch:
        if sample is None or sample[0] is None or sample[1] is None or sample[2] is None:
            continue
        num_block = num_block + sample[0].size(0)

    longest_sample = max(batch, key=func)
    if longest_sample is None or longest_sample[0] is None or longest_sample[1] is None or longest_sample[2] is None:
        print('longest_sample is None')
        return None

    max_seqlength = longest_sample[0].size(1)
    feats_size = longest_sample[0].size(2)
    
    batch_feats  = torch.zeros((num_block, max_seqlength, feats_size), dtype = torch.float32)
    batch_labels = torch.zeros((num_block, max_seqlength), dtype = torch.long) - 100
    batch_length = torch.LongTensor(num_block)
    
    iblock = 0
    for sample in batch:
        if sample is None or sample[0] is None or sample[1] is None or sample[2] is None:
            continue
            
        block_feats  = sample[0]       # [num_block, block_length, block_width]
        block_labels = sample[1]       # [num_block, block_length]
        block_length = sample[2]       # [num_block]
        
        block_size = block_feats.size(1)
        num_block = block_feats.size(0)
        
        batch_feats[iblock:iblock + num_block,  0:block_size, :] = block_feats
        batch_labels[iblock:iblock + num_block, 0:block_size]    = block_labels
        batch_length[iblock:iblock + num_block]                  = block_length

        iblock = iblock + num_block

    return batch_feats, batch_labels, batch_length

class BlockDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(BlockDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _block_collate_fn

    def shuffle(self):
        if self.batch_sampler is not None:
            self.batch_sampler.shuffle()
        if self.sampler is not None:
            self.sampler.shuffle()

class BlockBatchGenerator(object):
    def __init__(self, data):
        # input: 
        # block_feats:  (num_block, block_length, block_width)
        # block_labels: (num_block, block_length)
        # block_length: (num_block, 1)
        block_feats, block_labels, block_length = data
        
        self.num_read_block = 0
        self.num_block      = block_feats.size(0)
        self.block_rand_idx = np.random.permutation(self.num_block)
        
        self.block_feats  = block_feats
        self.block_labels = block_labels
        self.block_length = block_length
        
    def len(self):
        return self.num_block

    def is_empty(self):
        if self.num_read_block < self.num_block:
            return False
        else:
            return True

    def next_batch(self, batch_size):
        
        if self.is_empty():
            return  None, None, None

        if self.block_feats is None or self.block_labels is None or self.block_length is None:
            print('error: block_data is None')
            return None, None, None

        s = self.num_read_block
        e = min(s + batch_size, self.num_block)

        self.num_read_block += e - s

        indices      = torch.LongTensor(self.block_rand_idx[s:e])
        batch_feats  = self.block_feats.index_select(0, indices)
        batch_labels = self.block_labels.index_select(0, indices)
        batch_length = self.block_length.index_select(0, indices)
       
        return batch_feats, batch_labels, batch_length

class BlockUttDataLoader(BaseDataset):
    def __init__(self, data_args, speech_dir, noisy_dir, noise_dir, dataset = 'test', cmvn_file = None):
        super(BlockUttDataLoader, self).__init__(data_args, speech_dir, noisy_dir, noise_dir, dataset, cmvn_file, shuffle = False)

        self.data_args      = data_args
        self.block_length   = max(1, data_args.block_length)
        self.block_shift    = max(1, data_args.block_shift)
        self.block_width    = self.in_size

        self.feats  = []
        self.labels = []

        self.feats  += self.speech_feats
        self.labels += self.speech_labels

        self.feats  += self.noisy_feats
        self.labels += self.noisy_labels

        self.feats  += self.noise_feats
        self.labels += self.noise_labels

        self.feats  += self.babble_feats
        self.labels += self.babble_labels

        self.feats  += self.music_feats
        self.labels += self.music_labels

        self.num_utts = len(self.feats)

        self.num_loaded_utt = 0

    def len(self):
        return self.num_utts
    
    def is_empty(self):
        if self.num_loaded_utt < self.num_utts:
            return False
        else:
            return True

    def data_parser(self, feat_path, label):
        
        feat = self.parse_feat(feat_path, delta_order = self.delta_order, cmvn = self.cmvn, left_context_width = self.left_context_width, right_context_width = self.right_context_width)
        num_frame = feat.shape[0]
        
        if isinstance(label, str):
            label = self.parse_label(label, num_frame, given_label)
        else:
            label = self.parse_label(None, num_frame, label)
        
        return feat, label, num_frame

    def next_utt(self):
        
        if self.isempty():
            return None, None, None, None

        block_feats  = np.zeros(shape=(0, self.block_length, self.block_width), dtype = np.float32)
        block_labels = np.zeros(shape=(0, self.block_length), dtype = np.long)
        block_length = []

        utt_idx         = self.num_loaded_utt
        feat_key, feat_path    = self.feats[utt_idx]
        label_key, label_path  = self.labels[utt_idx]
        
        if feat_key.strip() != label_key.strip():
            print("feat key does not match label key")
            return None, None, None, None
        
        feats, labels, num_frame = self.data_parser(feat_path, label_path) 
        self.num_loaded_utt += 1

        block_frame_index = []
        for start in range(0, num_frame, self.block_shift):
            end = start + self.block_length
            if end < num_frame:
                sub_feats = feats[start:end, :][np.newaxis, :, :]
                sub_labels = labels[0, start:end]
                frames = end - start
            else:
                sub_feats = np.zeros(shape = (1, self.block_length, self.block_width), dtype=np.float32)
                sub_labels = np.zeros(shape = (1, self.block_length), dtype=np.int32) + self.ignore_label_id
                end = num_frame
                start = max(0, end - self.block_length)

                frames = end - start
                sub_feats[0, :frames, :] = feats[start:, :][np.newaxis, :, :]
                sub_labels[0, :frames]   = labels[0, start:]

            block_feats  = np.concatenate((block_feats, sub_feats), axis=0)
            block_labels = np.concatenate((block_labels, sub_labels), axis=0)
            block_length.append(frames)
            block_frame_index.append((start, end))
        
        block_feats     = torch.FloatTensor(block_feats)
        block_labels    = torch.LongTensor(block_labels)
        block_length    = torch.LongTensor(np.array(block_length))

        return block_feats, block_labels, block_length, block_frame_index