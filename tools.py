# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_GE2E -> tools
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/29 上午8:53
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
import librosa
import numpy as np
import scipy.fftpack

import constants as c


def mag_spec_extract(wav_file, process=False):
    '''
    Extract magnitude specturm / spectrogram
    process: False, Cut silence and fix length
    '''

    audio, sr = librosa.load(wav_file, c.sr)
    win_len = int(c.frame_win * c.sr)  # 400
    hop_len = int(c.frame_hop * c.sr)  # 160
    duration = c.num_frames * c.frame_hop + c.frame_win

    # Cut silence and fix length
    if process == True:
        audio, _ = librosa.effects.trim(audio, frame_length=win_len,
                                        hop_length=hop_len)
        length = int(duration * c.sr)
        audio = librosa.util.fix_length(audio, length)

    spec = librosa.stft(audio, n_fft=c.nfft,  # 1+n_fft//2, n_frames
                        hop_length=hop_len,
                        win_length=win_len)
    mag_spec = np.abs(spec)
    log_mag_spec = librosa.amplitude_to_db(mag_spec)

    return log_mag_spec, mag_spec


def fbank_extract(wav_file, process=False):
    '''
    Extract  fbank / log_mel_spec
    process: False, Cut silence and fix length
    '''
    _, mag_spec = mag_spec_extract(wav_file, process)
    pow_spec = np.square(mag_spec)
    mel_basic = librosa.filters.mel(c.sr, n_fft=c.nfft, n_mels=c.nmels)
    mel_spec = np.dot(mel_basic, pow_spec)  # n_mels, n_frame
    log_mel_spec = librosa.amplitude_to_db(mel_spec).T  # fbank  n_frame, n_mels

    return log_mel_spec, mel_spec


def mfcc_extract(wav_file, process=False):
    '''
    Extract  mfcc
    process: False, Cut silence and fix length
    '''
    log_mel_spec, _ = fbank_extract(wav_path)
    mfcc = scipy.fftpack.dct(log_mel_spec, axis=0)  # n_frame, n_mels
    return mfcc


if __name__ == '__main__':
    wav_path = r"/home/zcx/datasets/TIMIT/TIMIT_WAV/train_wav/DR1/FCJF0/SA1.WAV"
    # log_mag_spec, _ = mag_spec_extract(wav_path)
    # fbank, _ = fbank_extract(wav_path)
    mfcc = mfcc_extract(wav_path)
