# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_GE2E_Pytorch -> tools
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/25 上午10:34
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

import torch
from torch import nn
from torch.nn import functional as F

import constants as c


def get_centroids(embeddings):  # torch.Size([4, 10, 256]) --> torch.Size([4, 256])
    '''计算说话人（embeddings）的质心'''
    centroids = embeddings.mean(dim=1)  # torch.Size([4, 256])
    return centroids


def get_utterance_centroids(embeddings):  # torch.Size([4, 10, 256]) --> torch.Size([4, 10, 256])
    sum_embeddings = embeddings.sum(dim=1)  # torch.Size([4, 256])
    sum_embeddings = sum_embeddings.reshape(sum_embeddings.shape[0], 1,sum_embeddings.shape[-1])  # torch.Size([4, 1, 256])
    num_utterances = embeddings.shape[1] - 1  # 9
    utter_centroids = (sum_embeddings - embeddings) / num_utterances  # torch.Size([4, 10, 256])
    return utter_centroids


def get_cossim(embeddings, centroids):
    # embeddings  torch.Size([4, 10, 256])
    # centroids   torch.Size([4, 256])
    '''计算embeddings与centroids的余弦相似度'''
    num_utterances = embeddings.shape[1]  # 10
    utterance_centroids = get_utterance_centroids(embeddings)  # torch.Size([4, 10, 256]) --> torch.Size([4, 10, 256])
    utterance_centroids_flat = utterance_centroids.view(utterance_centroids.shape[0] * utterance_centroids.shape[1],-1)  # torch.Size([4*10, 256])
    embeddings_flat = embeddings.view(embeddings.shape[0] * num_utterances,-1)  # torch.Size([4*10, 256])
    cos_same = F.cosine_similarity(embeddings_flat,utterance_centroids_flat)  # 4*10

    centroids_expand = centroids.repeat(num_utterances * embeddings.shape[0],1)  # torch.Size([4*10*4, 256])
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1,embeddings.shape[0],1)  # torch.Size([40, 4, 256])
    embeddings_expand = embeddings_expand.view(embeddings_expand.shape[0] * embeddings_expand.shape[1],embeddings_expand.shape[-1])  # torch.Size([160, 256])
    cos_diff = F.cosine_similarity(embeddings_expand,centroids_expand)  # torch.Size([160])
    cos_diff = cos_diff.view(embeddings.size(0), num_utterances, centroids.size(0))  # torch.Size([4, 10, 4])

    # assign the cosine distance for same speakers to the proper idx
    cos_sim = cos_diff
    same_idx = list(range(embeddings.size(0)))  # [0, 1, 2, 3]
    cos_sim[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0],num_utterances)
    # torch.Size([4, 10, 4])

    cos_sim = cos_sim + 1e-6
    # torch.Size([4, 10, 4])
    return cos_sim


def calc_loss(sim_matrix):  # torch.Size([4, 10, 4])
    same_idx = list(range(sim_matrix.size(0)))  # [0,1,2,3]
    pos = sim_matrix[same_idx, :,same_idx]  # torch.Size([4, 10])
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()  # torch.Size([4, 10])
    per_embedding_loss = -1 * (pos - neg)  # torch.Size([4, 10])
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

