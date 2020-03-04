import random
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset
import python_speech_features


class WavDataset(Dataset):
    def __init__(self, utt2data, utt2label=None, label2int=None, need_aug=False, with_label=True, shuffle=True,
                 feat='mfcc'):
        self.utt2data = utt2data
        self.dataset_size = len(self.utt2data)
        self.shuffle = shuffle
        self.with_label = with_label
        self.utt2label = utt2label
        self.label2int = label2int
        self.need_aug = need_aug
        self.feat = feat

        if self.with_label:
            assert self.utt2label and self.label2int is not None, "utt2label must be provided in with_label model! "

        if shuffle:
            random.shuffle(self.utt2data)

    def __len__(self):
        return self.dataset_size

    def _transform_data(self, signal, sr):
        feat_func = getattr(python_speech_features, self.feat)
        if self.feat == 'mfcc':
            feat = feat_func(signal, sr, nfilt=64, numcep=64)
        else:
            feat = feat_func(signal, sr, nfilt=64)
        return feat.astype('float32')

    def augment(self, o_sig, sr, utt_label):
        return o_sig

    def __getitem__(self, sample_idx):
        idx = int(sample_idx)
        utt, filename = self.utt2data[idx]

        signal, sr = sf.read(filename)
        feat = self._transform_data(signal, sr)
        feat = torch.from_numpy(np.array(feat))

        if self.with_label:
            return utt, feat, int(self.label2int[self.utt2label[utt]])
        else:
            return utt, feat
