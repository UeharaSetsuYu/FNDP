from cProfile import label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import math
import torch



class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels): 
        device = features.device 
        features = F.normalize(features, dim=1) 
        similarity_matrix = torch.matmul(features, features.T) / self.temperature 
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError('Num of labels does not match num of features')

        mask = torch.eq(labels, labels.T).float().to(device) 
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask 
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach() 
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
 
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
 
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss

class ClusterContrastiveLoss(nn.Module):
    def __init__(self, class_num, temperature=0.5):
        super().__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def similarity(self, a, b): 
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.matmul(a.squeeze(1), b.squeeze(0).T)

    def mask_correlated_samples(self, N): 
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(False)
        for i in range(self.class_num):
            mask[i, self.class_num + i] = False
            mask[self.class_num + i, i] = False
        return mask

    def forward(self, q_i, q_j): 
        eps = 1e-10    
        p_i = q_i.sum(0)  # [C]
        p_i = p_i / (p_i.sum() + eps)
        p_j = q_j.sum(0)  # [C]
        p_j = p_j / (p_j.sum() + eps)
 
        ne_i = torch.log(torch.tensor(self.class_num, dtype=torch.float32, device=q_i.device)) \
               + (p_i * torch.log(p_i + eps)).sum()
        ne_j = torch.log(torch.tensor(self.class_num, dtype=torch.float32, device=q_i.device)) \
               + (p_j * torch.log(p_j + eps)).sum()
        entropy = ne_i + ne_j
 
        q_i = q_i.t()  # [C, B]
        q_j = q_j.t()  # [C, B]
        N = 2 * self.class_num
        q = torch.cat([q_i, q_j], dim=0)  # [2C, B]
 
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature  # [2C, 2C]
 
        sim_i_j = torch.diag(sim, self.class_num)  # [C]
        sim_j_i = torch.diag(sim, -self.class_num)  # [C]
        positive_clusters = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)  # [2C,1]
 
        mask = self.mask_correlated_samples(N).to(q.device)
        negative_clusters = sim[mask].reshape(N, -1)  # [2C, 2C-2]
 
        logits = torch.cat([positive_clusters, negative_clusters], dim=1)  # [2C, 1+(2C-2)]
        labels = torch.zeros(N, dtype=torch.long, device=q.device)  # 全部是正样本索引 0
        loss = self.criterion(logits, labels) / N
 
        return loss + entropy


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets): 
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)   
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class FeatureContrastiveLoss(nn.Module): 

    def __init__(self, temperature=0.5, eps=1e-8):
        super(FeatureContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, p, q): 
        device = p.device
        batch_size = p.size(0)
 
        p = F.normalize(p, dim=1)
        q = F.normalize(q, dim=1)
 
        sim_pq = torch.matmul(p, q.T) / self.temperature  # Cv_i 与 Cu_j
        sim_pp = torch.matmul(p, p.T) / self.temperature  # Cv_i 与 Cv_j
        sim_qq = torch.matmul(q, q.T) / self.temperature  # Cu_i 与 Cu_j
 
        exp_pq = torch.exp(sim_pq)
        exp_pp = torch.exp(sim_pp)
        exp_qq = torch.exp(sim_qq)

        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
 
        denom = exp_pp + exp_qq + exp_pq
        denom[mask] -= torch.exp(torch.tensor(1.0 / self.temperature, device=device))
 
        pos = torch.exp(torch.sum(p * q, dim=1) / self.temperature)
 
        loss = -torch.log(pos / (denom.sum(dim=1) + self.eps))
        loss = loss.mean()

        return loss
 

class Loss(nn.Module):
    def __init__(self, class_num, temperature_f=0.5, temperature_l=1.0):
        super(Loss, self).__init__()
        self.cluster_num = class_num
        self.pseudoAlignLoss = ClusterContrastiveLoss(class_num)
        self.mse = nn.MSELoss()
        self.BCELoss = nn.BCEWithLogitsLoss() 
        self.InstanceLevelCL = FeatureContrastiveLoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.MAE = nn.L1Loss()
        self.similarity = nn.CosineSimilarity(dim=2)
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.focal_loss = FocalLoss(gamma=5, alpha=0.05)
        self.Supervise_Loss = SupConLoss(temperature=0.07)


    def Supervise_Loss(self, feature, label):
        return self.Supervise_Loss(feature, label)
    def forward_align(self, q_i, q_j):
        return self.pseudoAlignLoss(q_i, q_j)

    def forward_mse(self, x, y):
        return self.mse(x, y)

    def forward_BCE(self, x, y):
        return self.BCELoss(x, y)

    def forward_Consis(self, x, y):
        return self.Consis_loss(x, y)

    def Cluster_num(self):
        return self.cluster_num

    def forward_ILCL(self, p, q):
        return self.InstanceLevelCL(p, q)

    def forward_CrossEntropy(self, logit, label):
        return self.crossentropy(logit, label)

    def forward_Soft_CrossEntropy(self, logit, label):
        return self.soft_cross_entropy(logit, label)

    def forward_Entropy(self, x, eps=1e-8):
        probs = F.softmax(x, dim=1)
        return - (probs * probs.log()).sum(dim=1).mean()

    def forward_Focal_CE(self, p, q):
        return self.focal_loss(p, q)

    def soft_cross_entropy(self, pred, soft_targets):
        log_prob = F.log_softmax(pred, dim=1)
        return -(soft_targets * log_prob).sum(dim=1).mean()

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.cluster_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.cluster_num)
        sim_j_i = torch.diag(sim, -self.cluster_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_MI(self, x, y):
        return instance_contrastive_Loss(x, y)


''' Ablation W/o Adversarial '''
# MI-based
import torch

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)
    '''unsqueeze(i)的目的是进行广播计算，会在第i维多一维数据出来'''
    '''(256,128,1) * (256,1,128) = (256,128,128)，三维矩阵结构(batch, height, width)，(h, w)进行二维矩阵相乘计算，得到batch维结果'''
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise  对称化
    p_i_j = p_i_j / (p_i_j.sum() + 1e-8)  # normalise    得到标准化的联合概率分布
    # if p_i_j.sum() == 0: print("-------------------------------------------Warning!!!!------------------------------------")
    return p_i_j

def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=1e-8):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))
    '''view(i, j)的作用是将矩阵维度重置为符合规定的(i,j)，类似reshape'''
    '''expand(k, k)可将矩阵维度为1的方向进行扩展'''
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)  # 行和    提取出边缘概率
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # 列和
    '''以下三行代码的作用是：防止数值过小：若小于EPS，则用EPS代替数据；反之则使用原数据。torch.where类似三目运算符'''
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))  # 最终的损失计算公式

    loss = loss.sum()   # 得到总和，即为icl的损失

    return loss



 