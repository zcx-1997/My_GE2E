# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_GE2E_Pytorch -> constants
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/26 下午5:47
@Description        ：
==================================================
"""

# --dataroot:  default='./voxceleb'
# --test-pairs-path: default='./voxceleb/voxceleb1_test3.txt'
# --log-dir: default='./data/pytorch_speaker_logs'
# --resume: help='path to latest checkpoint (default: none)'
# --start-epoch： help='manual epoch number (useful on restarts)'
# --epochs：help='number of epochs to train (default: 10)'
# --embedding-size： default=512
# --batch-size：default=512, help='input batch size for training (default: 128)'
# --test-batch-size: help='input batch size for testing (default: 64)'
# --test-input-per-file: help='input sample per file for testing (default: 8)'
# --n-triplets:default=1000000,help='how many triplets will generate from the dataset'
# --margin:default=0.1,help='the margin value for the triplet loss function (default: 1.0'
# --min-softmax-epoch:help='minimum epoch for initial parameter using softmax (default: 2'
# --loss-ratio:help='the ratio softmax loss - triplet loss (default: 2.0'
# --lr:default=0.1,help='learning rate (default: 0.125)'
# --lr-decay:default=1e-4,help='learning rate decay ratio (default: 1e-4'
# --wd:help='weight decay (default: 0.0)'
# --optimizer:help='The optimizer to use (default: Adagrad)'
# --no-cuda:help='enables CUDA training'
# --gpu-id:default='3', help='id(s) for CUDA_VISIBLE_DEVICES'
# --seed:default=0,help='random seed (default: 0)'
# --log-interval:help='how many batches to wait before logging training status'
# --mfb:help='start from MFB file'
# --makemfb:help='need to make mfb file'

seed = 0
checkpoints = r'checkpoints'

# data
nmels = 40
batch_size = 4

# model
lstm_num_hidden = 768
lstm_num_layer = 3
fc_num_hidden = 256

# trian
epochs = 500
lr = 0.01

# test
test_epochs = 500
test_num_utters = 10








BATCH_SIZE = 32        #must be even
TRIPLET_PER_BATCH = 3

SAVE_PER_EPOCHS = 200
TEST_PER_EPOCHS = 200
CANDIDATES_PER_BATCH = 640       # 18s per batch
TEST_NEGATIVE_No = 99


NUM_FRAMES = 160   # 299 - 16*2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
ALPHA = 0.2
HIST_TABLE_SIZE = 10
NUM_SPEAKERS = 251
DATA_STACK_SIZE = 10

CHECKPOINT_FOLDER = 'checkpoints'
BEST_CHECKPOINT_FOLDER = 'best_checkpoint'
PRE_CHECKPOINT_FOLDER = 'pretraining_checkpoints'
GRU_CHECKPOINT_FOLDER = 'gru_checkpoints'

LOSS_LOG= CHECKPOINT_FOLDER + '/losses.txt'
TEST_LOG= CHECKPOINT_FOLDER + '/acc_eer.txt'

PRE_TRAIN = False

COMBINE_MODEL = False