import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.SE3 import *


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """ 
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence 
        - weight_threshold: float,    clips points with weight below threshold裁剪权重低于阈值的点
    Output:
        - R, t 
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud计算点云中所有点的平均位置，这个平均位置就是点云的质心
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean对点云中的每个点，将其坐标减去质心的坐标。这个过程使得点云的质心移动到坐标系的原点
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix  构造权重协方差矩阵
    # 将一个张量中的元素作为对角线元素，其余元素置零，生成一个指定维度的方阵
    Weight = torch.diag_embed(weights)
    # 将Am后两个维度颠倒，即（1600，3，20）H=AT*W*B
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation，
    # 对H进行SVD分解
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    # 左右奇异矩阵转置相乘后的行列式值 shape：1600个数字
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    # (1600, 3, 3)
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV #将 delta_UV 的值赋给 eye 张量的最后一个位置，即最后一个对角线元素，作为缩放比例
    R = Vt @ eye @ U.permute(0, 2, 1)#使用计算得到的左右奇异矩阵以及带有缩放比例的单位矩阵，计算刚体变换矩阵 R。
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)#应用刚体变换矩阵 R 对 centroid_A 进行变换，并计算得到变换后的结果 t。
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    # 将旋转矩阵R和平移矩阵t组成一个4 * 4矩阵
    return integrate_trans(R, t)


def knn(x, k, ignore_self=False, normalized=True):
    """ find feature space knn neighbor of x 
    Input:
        - x:       [bs, num_corr, num_channels],  input features
        - k:       
        - ignore_self:  True/False, return knn include self or not.
        - normalized:   True/False, if the feature x normalized.
    Output:
        - idx:     [bs, num_corr, k], the indices of knn neighbors
    """
    inner = 2 * torch.matmul(x, x.transpose(2, 1))
    if normalized:
        pairwise_distance = 2 - inner
    else:
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = xx - inner + xx.transpose(2, 1)

    if ignore_self is False:
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    else:
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


class EdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, k, idx=None):
        super(EdgeConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.idx = idx
        self.conv = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [bs, in_dim, N]
        bs = x.shape[0]
        num_corr = x.shape[2]
        device = x.device

        # if self.idx is None:
        self.idx = knn(x.permute(0,2,1), self.k, normalized=False)

        idx_base = torch.arange(0, bs, device=device).view(-1, 1, 1) * num_corr
        idx = self.idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()
        features = x.view(bs * num_corr, -1)[idx, :]
        features = features.view(bs, num_corr, self.k, self.in_dim)
        x = x.view(bs, num_corr, 1, self.in_dim).repeat(1, 1, self.k, 1)

        features = torch.cat([features - x, x], dim=3).permute(0, 3, 1, 2).contiguous()

        output = self.conv(features)
        output = output.max(dim=-1, keepdim=False)[0]
        return output


class ContextNormalization(nn.Module):
    def __init__(self):
        super(ContextNormalization, self).__init__()

    def forward(self, x):
        var_eps = 1e-3
        mean = torch.mean(x, 2, keepdim=True)
        variance = torch.var(x, 2, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + var_eps)
        return x


class PointCN(nn.Module):
    def __init__(self, in_dim=6, num_layers=6, num_channels=128, act_pos='post'):
        super(PointCN, self).__init__()
        assert act_pos == 'pre' or act_pos == 'post'

        modules = [nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)]
        for i in range(num_layers):
            if act_pos == 'pre':
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
            else:
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        features = self.encoder(x)
        return features
