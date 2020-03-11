import numpy as np
import random
import soundfile
from voxceleb1.dataset.fftfilt import fftfilt


class NoiseAugment:
    def __init__(self, noise_list):
        self.noise_list = [x.strip() for x in open(noise_list)]

    def __call__(self, y, sr, snr=0):
        noise_path = random.choice(self.noise_list)
        noise, sr_ = soundfile.read(noise_path)
        assert sr_ == sr and noise.ndim == 1
        # pad repeatedly or truncate noise to the length of y
        speech_len = y.shape[-1]
        if len(noise) < speech_len:
            repeat = speech_len // len(noise) + 1
            noise = np.concatenate([noise] * repeat)
        offset = random.randint(0, len(noise) - speech_len)
        noise = noise[offset:offset + speech_len]
        # mix noise and y by snr
        sigma = np.sqrt(10 ** (-snr / 10))
        y_aug = y + sigma * noise
        # avoid too large volume
        y_aug = y_aug / (np.abs(y_aug).max() + 1e-15)
        return y_aug


class RIRAugment:
    def __init__(self, rir_list):
        self.rir_list = [x.strip() for x in open(rir_list)]

    def __call__(self, y, sr, nfft=None):
        rir_path = random.choice(self.rir_list)
        # rir is the impluse response
        rir, sr_ = soundfile.read(rir_path)
        assert sr_ == sr and rir.ndim == 1
        y_aug = fftfilt(rir, y, nfft)
        y_aug = y_aug / (np.abs(y_aug).max() + 1e-15)
        return y_aug


if __name__ == '__main__':
    random.seed(0)
    wav_path = '/data1/linqj/King-ASR-111_split/379b1_funnythings_001_0004.wav'
    y, sr = soundfile.read(wav_path)

    noise_aug = NoiseAugment('../data/musan_noise/wav_list')
    y_aug = noise_aug(y, sr, snr=20)
    soundfile.write('noise_aug.wav', y_aug, sr)

    rir_aug = RIRAugment('../data/simu_rirs/wav_list')
    y_aug = rir_aug(y, sr)
    soundfile.write('rir_aug.wav', y_aug, sr)
