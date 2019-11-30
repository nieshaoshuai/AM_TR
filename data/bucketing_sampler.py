from torch.utils.data.sampler import Sampler
import numpy as np
from data.data_loader_new import SequentialDataset
from audio.audio_io import load_audio, load_mat

from collections import defaultdict

import random

class SequentialDatasetWithLength(SequentialDataset):
    def __init__(self, *args, **kwargs):
        """
        SpectrogramDataset that splits utterances into buckets based on their length.
        Bucketing is done via numpy's histogram method.
        Used by BucketingSampler to sample utterances from the same bin.
        """
        super(SequentialDatasetWithLength, self).__init__(*args, **kwargs)

        self.audio_lengths = [self.FileSize(path[0]) for path in self.utt_ids]
        #audio_lengths = [self.nSamples(path) for path in self.utt_ids]

        hist, bin_edges = np.histogram(self.audio_lengths, bins="auto")
        audio_samples_indices = np.digitize(self.audio_lengths, bins=bin_edges)

        self.bins_to_samples = defaultdict(list)
        for idx, bin_id in enumerate(audio_samples_indices):
            self.bins_to_samples[bin_id].append(idx)
        #random.shuffle(self.bins_to_samples)

class BucketingSequentialSampler(Sampler):
    def __init__(self, data_source):
        """
        Samples from a dataset that has been bucketed into bins of similar sized sequences to reduce
        memory overhead.
        :param data_source: The dataset to be sampled from
        """
        super(BucketingSequentialSampler, self).__init__(data_source)
        self.data_source = data_source
        assert hasattr(self.data_source, 'bins_to_samples')

    def ShuffleSampler(self):
        random.shuffle(self.data_source.bins_to_samples)

    def __iter__(self):
        for bin, sample_idx in self.data_source.bins_to_samples.items():
            np.random.shuffle(sample_idx)
            for s in sample_idx:
                yield s

    def __len__(self):
        return len(self.data_source)
        

class BucketingBatchSequentialSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingBatchSequentialSampler, self).__init__(data_source)
        self.data_source = data_source
        assert hasattr(self.data_source, 'bins_to_samples')
        
        self.batch_size = batch_size
        self.bins = self.build_bins()

    def __iter__(self):
        for ids in self.bins:
            #np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)
        
    def build_bins(self):
        ids = []
        for bin, sample_idx in self.data_source.bins_to_samples.items():
            ids.extend(sample_idx)
        bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        #np.random.shuffle(bins)
        return bins