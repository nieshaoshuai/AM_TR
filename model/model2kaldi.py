import os
import numpy as np
import torch

def write2kaldimodel(net_file_name, out_file_name, feat_size, cmvn_file=None):
    cmvn = None
    if os.path.exists(cmvn_file):
        cmvn = np.load(cmvn_file)
        cmvn_size = np.shape(cmvn)[1]
    checkpoint = torch.load(net_file_name)
    state_dict = 'state_dict'
    if state_dict in checkpoint:
        net_dict = checkpoint[state_dict]
        with open(out_file_name, 'w') as f:
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
            for k, v in net_dict.items():
                if v.is_cuda:
                    v = v.cpu()
                if "weight" in k:
                    in_size, out_size = v.shape
                    f.write('<AffineTransform> {0} {1}\n'.format(in_size, out_size))
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
                elif "bias" in k:
                    bias_size = len(v)
                    f.write(' [ ')
                    for i in range(bias_size):
                        f.write('%.9f ' % v[i])
                    f.write(']\n')
                    f.write('<ReLU> {0} {1}\n'.format(bias_size, bias_size))
            f.write('<output> {0} {1}\n'.format(bias_size, bias_size))
    else:
        print ('{0} is not right, please check it'.format(net_file_name))  