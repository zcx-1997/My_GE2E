# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_GE2E_Pytorch -> train
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/17 上午10:14
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
import time

import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import constants
import constants as c
from data_loader import Vox1Dataset,Vox1Dataset_train_100,TIMITDataset
from model import NetEmbedder, GE2ELoss
from tools import get_centroids, get_cossim

#计算准确率
def sum_right(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum())

def accuracy(y_hat, y):
    """计算准确率。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum()) / len(y)


def train(device):
    checkpoint_dir = r'checkpoints'
    log_file = os.path.join(checkpoint_dir,'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("load data...")
    # train_db = Vox1Dataset()
    # train_db = Vox1Dataset_train_100()
    train_db = TIMITDataset()
    train_loader = DataLoader(train_db, batch_size=c.batch_size, shuffle=True,drop_last=True)

    net = NetEmbedder()
    net = net.to(device)

    ge2e_loss = GE2ELoss(device)
    # criterion = nn.CrossEntropyLoss()
    opt = optim.SGD([{"params":net.parameters()},
                     {"params":ge2e_loss.parameters()}], c.lr)
    writer = SummaryWriter()

    for epoch in range(c.epochs):
        net.train()
        total_loss = 0
        for step_id, x in enumerate(train_loader):
            x = x.to(device)
            x = x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])

            # 两个索引列表，互为索引，用来将原始样本 x 与模型输出 y 一一对应
            perm = random.sample(range(0, x.shape[0] * x.shape[1]), x.shape[0] * x.shape[1])
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            x = x[perm]
            opt.zero_grad()

            embeddings = net(x)
            embeddings = embeddings[unperm]

            loss = ge2e_loss(embeddings)
            loss.backward()
            opt.step()

            writer.add_scalar("train_loss", loss, epoch * step_id)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("train_avg_loss", avg_loss, epoch)

        if (epoch + 1) % 1 == 0:
            message = "Train: epoch{}, avg_loss={:.4f}, loss={:.4f}, time={}".format(epoch + 1,avg_loss,loss,time.ctime())
            print(message)

            if checkpoint_dir is not None:
                with open(log_file, 'a') as f:
                    f.write(message+'\n')

        if checkpoint_dir is not None and (epoch + 1) % 100 == 0:
            net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(epoch + 1) + ".pth"
            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
            torch.save(net.state_dict(), ckpt_model_path)
            net.to(device).train()
    writer.close()

    # save model
    net.eval().cpu()
    save_model_filename = "final_epoch_" + str(epoch + 1) + ".model"
    save_model_path = os.path.join(checkpoint_dir, save_model_filename)
    torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

def test(model_path):
    test_db = TIMITDataset(train=False)
    test_loader = DataLoader(test_db, batch_size=c.batch_size, shuffle=True,drop_last=True)

    embedder_net = NetEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    avg_EER = 0
    epochs = 10
    for epoch in range(epochs):  # epochs: 10
        batch_avg_EER = 0
        for step_id, x in enumerate(test_loader):  # mel_db_batch torch.Size([4, 10, 160, 40])
            assert x.shape[1] % 2 == 0
            enroll_utters, veri_utters = torch.split(x, int(x.shape[1]/2), dim=1)
            # enrollment_batch torch.Size([4, 5, 160, 40]), verification_batch torch.Size([4, 5, 160, 40])

            enroll_utters = torch.reshape(enroll_utters, (enroll_utters.shape[0]*enroll_utters.shape[1], enroll_utters.shape[2],enroll_utters.shape[3]))
            # torch.Size([4*5, 160, 40])
            veri_utters = torch.reshape(veri_utters,(veri_utters.shape[0]*veri_utters.shape[1],veri_utters.shape[2],veri_utters.shape[3]))
            # torch.Size([4*5, 160, 40])

            perm = random.sample(range(0, veri_utters.shape[0]), veri_utters.shape[0])
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            veri_utters = veri_utters[perm]

            enroll_embeddings = embedder_net(enroll_utters)  # torch.Size([4*5, 256])
            veri_embeddings = embedder_net(veri_utters)  # torch.Size([4*5, 256])
            veri_embeddings = veri_embeddings[unperm]

            enroll_embeddings = enroll_embeddings.reshape(enroll_embeddings.shape[0],enroll_embeddings.shape[1],enroll_embeddings.shape[2])
            veri_embeddings = veri_embeddings.reshape(veri_embeddings.shape[0],veri_embeddings.shape[1],veri_embeddings.shape[2])  # torch.Size([4, 5, 256])


            enroll_centroids = get_centroids(enroll_embeddings)  # torch.Size([4, 5, 256]) --> torch.Size([4, 256])

            sim_matrix = get_cossim(veri_embeddings, enroll_centroids)  # torch.Size([4, 5, 4])

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0

            for thres in [0.01 * i + 0.5 for i in range(50)]:  # [0.5 ~ 1]
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :,i].float().sum() for i in range(int(c.batch_size))]) / (c.batch_size - 1.0) / (float(c.test_num_utters / 2)) / c.batch_size)
                FRR = (sum([c.test_num_utters / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(c.batch_size))]) / (float(c.test_num_utters / 2)) / c.batch_size)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
            avg_EER += batch_avg_EER / (step_id + 1)
    avg_EER = avg_EER / c.test_epochs
    print("\n EER across {0} epochs: {1:.4f}".format(c.test_epochs, avg_EER))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on: ", device)
    train(device)

    # model_path = r'checkpoints/final_epoch_1000.model'
    # acc = test(model_path)
    # print(acc)  # 0.873800.

