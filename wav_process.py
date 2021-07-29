# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_GE2E -> wav_process
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/29 上午8:52
@Description        ：
                    _ooOoo_    
                   o8888888o    
                   88" . "88    
                   (| -_- |)    
                    O\ = /O    
                ____/`---'\____    
                 .' \\| |// `.    
               / \\||| : |||// \    
             / _||||| -:- |||||- \    
               | | \\\ - /// | |    
             | \_| ''\---/'' | |    
              \ .-\__ `-` ___/-. /    
           ___`. .' /--.--\ `. . __    
        ."" '< `.___\_<|>_/___.' >'"".    
       | | : `- \`.;`\ _ /`;.`/ - ` : | |    
         \ \ `-. \_ __\ /__ _/ .-` / /    
 ======`-.____`-.___\_____/___.-`____.-'======    
                    `=---='    
 .............................................    
              佛祖保佑             永无BUG
==================================================
"""
import os
import glob
import librosa
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader

import constants as c

def extractFeatures(src_path, tar_path):
    """
    Extract features, the features are saved as numpy file.
    Each partial utterance is splitted by voice detection using DB
    the first and the last 180 frames from each partial utterance are saved.
    """



    print("start text independent utterance feature extraction")
    train_path = os.path.join(tar_path, 'train_tisv')
    test_path = os.path.join(tar_path, 'test_tisv')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    utter_len = (c.num_frames * c.frame_hop + c.frame_win) * c.sr

    spk_list = glob.glob(os.path.dirname(src_path))
    total_num_spks = len(spk_list)
    # split total data 90% train and 10% test
    train_num_spks = (total_num_spks // 10) * 9
    test_num_spks = total_num_spks - train_num_spks
    print("total speaker number : %d" % total_num_spks)
    print("train : %d, test : %d" % (train_num_spks, test_num_spks))

    for i, spk_dir in enumerate(spk_list):
        # eg. spk_dir = '/home/zcx/datasets/TIMIT/TIMIT_WAV/train_wav/DR3/MCAL0'
        print("%dth speaker processing..." % i)
        spk_spec = []
        for wav_name in os.listdir(spk_dir):  # 'SI1768.WAV'
            wav_path = os.path.join(spk_dir, wav_name)
            audio, sr = librosa.load(wav_path, sr=c.sr)
            # voice activity detection, return [[start end] [start end] ...]
            intervals = librosa.effects.split(audio, top_db=30)
            for interval in intervals:
                if interval[1] - interval[0] >= utter_len:
                    audio = audio[interval[0]:interval[1]]

                    S = librosa.stft(audio, n_fft=c.nfft,
                                     win_length=int(c.frame_win * sr),
                                     hop_length=int(c.frame_hop * sr))
                    S = np.abs(S) ** 2  # abs --> square
                    mel_basis = librosa.filters.mel(sr=sr,
                                                    n_fft=c.nfft,
                                                    n_mels=c.nmels)
                    # log mel spectrogram of utterances
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)
                    spk_spec.append(S[:, :c.num_frames])
                    spk_spec.append(S[:, -c.num_frames:])

        spk_spec = np.array(spk_spec)
        print(spk_spec.shape)

        # # save spectrogram as numpy file
        # if i<train_num_spks:
        #     np.save(os.path.join(train_path, "%d.npy"%i), spk_spec)
        # else:
        #     np.save(os.path.join(test_path, "%d.npy"%(i)), spk_spec)


if __name__ == '__main__':
    src_path = '/home/zcx/datasets/TIMIT/TIMIT_WAV/*/*/*/*.wav'
    tar_path = './data'

    extractFeatures(src_path, tar_path)  # error

    wav_paths = glob.glob(src_path)
    utter_len = (c.frame_hop * c.num_frames + c.frame_win) * c.sr
    # print(wav_paths)
    audio, sr = librosa.load(wav_paths[3], c.sr)
    intervals = librosa.effects.split(audio, top_db=30)
    print(intervals)
    spe = []
    for interval in intervals:
        if interval[1] - interval[0] >= utter_len:
            audio = audio[interval[0]:interval[1]]
            S = librosa.stft(audio, n_fft=c.nfft,
                             win_length=int(c.frame_win * c.sr),
                             hop_length=int(c.frame_hop * c.sr))
            S = np.abs(S) ** 2  # abs --> square
            mel_basis = librosa.filters.mel(sr=c.sr,
                                            n_fft=c.nfft,
                                            n_mels=c.nmels)
            # log mel spectrogram of utterances
            S = np.log10(np.dot(mel_basis, S) + 1e-6)
            print(S)
            print(S.shape)
            spe.append(S[:, :160])
            spe.append(S[:, -160:])
            print(spe)
            print(len(spe))
            print(len(spe[0]))
            print(len(spe[0][0]))
