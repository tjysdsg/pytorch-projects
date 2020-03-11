import numpy as np
import random
import soundfile
from torch.utils.data import Dataset
from dep.python_speech_features.python_speech_features import logfbank
from voxceleb1.dataset.data_augment import NoiseAugment, RIRAugment


class FbankDataset(Dataset):
    def __init__(self, wav_scp, utt2spk=None, spk2int=None, fbank_kwargs=None, padding="wrap", cmn=False, aug=False):
        """
        Params:
            wav_scp         - <utt> <wavpath>
            utt2spk         - <utt> <spk>
            spk2int         - <spk> <int>
            fbank_kwargs    - config of fbank features
            padding         - "wrap" or "constant"(zeros), for feature truncation, 
                              effective only if waveform length is less than truncated length.
            cmn             - whether perform mean normalization for feats.
        """
        if fbank_kwargs is None:
            fbank_kwargs = {}
        self.utt2wavpath = {x.split()[0]: x.split()[1] for x in open(wav_scp)}
        self.utt2label = self.init_label(utt2spk, spk2int)
        self.utts = sorted(list(self.utt2wavpath.keys()))
        self.fbank_kwargs = fbank_kwargs
        self.padding = 'wrap' if padding == "wrap" else 'constant'
        self.cmn = cmn
        self.len = len(self.utts)
        self.aug = aug
        if self.aug:
            self.music_aug = NoiseAugment('data/musan_music/wav_list')
            self.noise_aug = NoiseAugment('data/musan_noise/wav_list')
            self.babble_aug = NoiseAugment('data/musan_speech/wav_list')
            self.rir_aug = RIRAugment('data/simu_rirs/wav_list')

    def init_label(self, utt2spk, spk2int=None):
        """
        Transform speaker to int, for example: A --> 1, B --> 2. 
        Map utt with integer spk.
        """
        if utt2spk is None:
            return None
        utt2spk = {x.split()[0]: x.split()[1] for x in open(utt2spk)}
        if spk2int is None:
            spks = sorted(set(utt2spk.values()))
            spk2int = {spk: i for i, spk in enumerate(spks)}
        else:
            spk2int = {x.split()[0]: int(x.split()[1]) for x in open(spk2int)}
        utt2label = {utt: spk2int[spk] for utt, spk in utt2spk.items()}
        return utt2label

    def trun_wav(self, y, tlen, padding):
        """
        Truncation, zero padding or wrap padding for waveform.
        """
        # no need for truncation or padding
        if tlen is None:
            return y
        n = len(y)
        # needs truncation
        if n > tlen:
            offset = random.randint(0, n - tlen)
            y = y[offset:offset + tlen]
            return y
        # needs padding (zero/repeat padding)
        y = np.pad(y, (0, tlen - n), mode=padding)
        return y

    def aug_wav(self, y, sr):
        aug_list = ['clean', 'clean', 'clean',
                    'music', 'noise', 'babble']
        aug_type = random.choice(aug_list)

        snr = random.randint(5, 20)
        if aug_type == 'clean':
            return y
        elif aug_type == 'music':
            return self.music_aug(y, sr, snr)
        elif aug_type == 'noise':
            return self.noise_aug(y, sr, snr)
        elif aug_type == 'babble':
            # repeat for multiple times to simulate the babble noise
            for i in range(random.randint(3, 6)):
                y = self.babble_aug(y, sr, snr)
            return y
        elif aug_type == 'rir':
            return self.rir_aug(y, sr)
        else:
            raise AssertionError

    def extract_fbank(self, y, sr, fbank_kwargs, cmn=False):
        feat = logfbank(y, sr, winfunc=np.hamming, **fbank_kwargs)
        if cmn:
            feat -= feat.mean(axis=0, keepdims=True)
        return feat.astype('float32')

    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError

        utt = self.utts[index]
        y, sr = soundfile.read(self.utt2wavpath[utt])
        y = self.trun_wav(y, tlen, self.padding)

        if self.aug:
            y = self.aug_wav(y, sr)

        feat = self.extract_fbank(y, sr, fbank_kwargs=self.fbank_kwargs, cmn=self.cmn)
        if self.utt2label is None:
            return utt, feat
        label = self.utt2label[utt]
        return utt, feat, label

    def __len__(self):
        return self.len
