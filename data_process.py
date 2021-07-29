# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_GE2E -> data_process
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/29 下午5:05
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
import numpy as np
import constants as c

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
        Need : utterance data set (VTCK)
    """




    print("start text independent utterance feature extraction")
    tar_path = './data'
    train_path = os.path.join(tar_path, 'train_t')
    test_path = os.path.join(tar_path, 'test_t')
    os.makedirs(train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (c.num_frames * c.frame_hop + c.frame_win) * c.sr    # lower bound of utterance length

    audio_path = glob.glob(
        os.path.dirname('/home/zcx/datasets/TIMIT/TIMIT_WAV/*/*/*/*.wav'))
    total_speaker_num = len(audio_path)
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))

    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..."%i)
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.WAV':
                utter_path = os.path.join(folder, utter_name)         # path of each utterance
                utter, sr = librosa.core.load(utter_path, c.sr)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection
                # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                # for vctk dataset use top_db=100
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=c.nfft,
                                              win_length=int(c.frame_win * sr), hop_length=int(c.frame_hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=c.sr, n_fft=c.nfft, n_mels=c.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :c.num_frames])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -c.num_frames:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(train_path, "%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(test_path, "%d.npy"%(i-train_speaker_num)), utterances_spec)

if __name__ == '__main__':
    save_spectrogram_tisv()