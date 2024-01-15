import torch
from common import knn, rigid_transform_3d
from utils.SE3 import transform
import numpy as np



class Matcher():
    def __init__(self,
                 inlier_threshold=0.10,
                 num_node='all',
                 use_mutual=True,
                 d_thre=0.1,
                 num_iterations=10,
                 ratio=0.2,
                 nms_radius=0.1,
                 max_points=8000,
                 k1=30,
                 k2=20,
                 select_scene=None,
                 ):
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.d_thre = d_thre
        self.num_iterations = num_iterations  # maximum iteration of power iteration algorithm
        self.ratio = ratio # the maximum ratio of seeds.
        self.max_points = max_points
        self.nms_radius = nms_radius
        self.k1 = k1
        self.k2 = k2
    # 通过置信度大小及匹配对距离半径得到种子点（一些种子点置信度与较高置信度可能较小，但是它离的较高置信度那些点都比较远，也会被选择。种子要分散开）
    def pick_seeds(self, dists, scores, R, max_num):
        """
        Select seeding points using Non Maximum Suppression非极大抑制. (here we only support bs=1)
        Input:
            - dists:       [bs, num_corr, num_corr] src keypoints distance matrix关键点距离矩阵
            - scores:      [bs, num_corr]     initial confidence of each correspondence初始化每个配准置信度
            - R:           float              radius of nms
            - max_num:     int                maximum number of returned seeds种子最大个数
        Output:
            - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences返回原始配准的index

        """
        assert scores.shape[0] == 1

        # parallel Non Maximum Suppression (more efficient)
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        # 找到置信度最大和一些置信度比一些匹配correspond比较高且距离比那些更高的置信度点远（这些匹配对置信度如果没有那些高，但是离那些高置信度的远）
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        # 在score_relation最后一个维度上执行最小值操作，取得最小值的第一个元素，也就是最小值本身，并转化为浮点数，只有一行全为T才是被挑选
        is_local_max = score_relation.min(-1)[0].float()
        #
        score_local_max = scores * is_local_max
        # 返回被挑选种子index（按置信度），
        sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

        # max_num = scores.shape[1]
        # 选择前max_num个种子点index
        return_idx = sorted_score[:, 0: max_num].detach()

        return return_idx
    # 计算所有种子集的刚性变换矩阵，并通过求取全部source点经刚性变化后预测点与实际target距离，最后根据阈值筛选得到内点最多那个刚性矩阵
    def cal_seed_trans(self, seeds, SC2_measure, src_keypts, tgt_keypts):
        """
        Calculate the transformation for each seeding correspondences.
        Input:
            - seeds:         [bs, num_seeds]              the index to the seeding correspondence
            - SC2_measure: [bs, num_corr, num_channels]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
        Output: leading eigenvector
            - final_trans:       [bs, 4, 4]             best transformation matrix (after post refinement) for each batch.
        """
        bs, num_corr, num_channels = SC2_measure.shape[0], SC2_measure.shape[1], SC2_measure.shape[2]
        k1 = self.k1
        k2 = self.k2
        
        if k1 > num_channels:
            k1 = 4
            k2 = 4

        #################################
        # The first stage consensus set sampling
        # Finding the k1 nearest neighbors around each seed
        # 第一阶段共识集抽样
        # 找到每个种子周围的 k1 最近邻
        #################################
        # 按行进行对每个种子和其他correspond的相似度（有几个相同的匹配对）大小进行排序返回index
        sorted_score = torch.argsort(SC2_measure, dim=2, descending=True)
        knn_idx = sorted_score[:, :, 0: k1]
        # 大小进行排序返回值
        sorted_value, _ = torch.sort(SC2_measure, dim=2, descending=True)
        # 对每个种子找到的30个correspond index：knn_idx变形为【1，48000】
        idx_tmp = knn_idx.contiguous().view([bs, -1])
        # 增加一个维度
        idx_tmp = idx_tmp[:, :, None]
        # 变形为【1，48000，3】，广播机制
        idx_tmp = idx_tmp.expand(-1, -1, 3)

        #################################
        # construct the local SC2 measure of each consensus subset obtained in the first stage.
        # 构造第一阶段获得的每个共识子集的局部SC2度量。
        # #################################
        # 得到每个种子集的k个点的坐标,包括sorce及target point，并按种子分开，[bs, num_seeds, k, 3]
        src_knn = src_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])  # [bs, num_seeds, k, 3]
        tgt_knn = tgt_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])
        # 一个种子集中每个点与其他点坐标先算出距离
        src_dist = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_SC_measure = (cross_dist < self.d_thre).float()
        # 作者以每个共识集（种子集）的第一个correspond作为种子标准（）计算SC2矩阵，算出来的时每个种子的第一个correspond
        # 与其他correspond有几个共识correspond，可以看到SC2的第一个都是30，那是他与自己的共识点数，
        # 不忽略自己是因为，我们不需要剔除它 【1，1600，1，30】

        local_SC2_measure = torch.matmul(local_hard_SC_measure[:, :, :1, :], local_hard_SC_measure)

        #################################
        # perform second stage consensus set sampling
        #################################
        # 对local_SC2_measure进行排序，返回索引及前K2的索引，【1，1600，1，20】
        sorted_score = torch.argsort(local_SC2_measure, dim=3, descending=True)
        knn_idx_fine = sorted_score[:, :, :, 0: k2]

        #################################
        # construct the soft SC2 matrix of the consensus set构造共识集的软 SC2 矩阵
        #################################
        # 种子数量
        num = knn_idx_fine.shape[1]
        # shape：【1，1600，1，20】--->[1,1600,20]--->[1,1600,20,1]--->[1,1600,20,3]
        knn_idx_fine = knn_idx_fine.contiguous().view([bs, num, -1])[:, :, :, None]
        knn_idx_fine = knn_idx_fine.expand(-1, -1, -1, 3)
        # 得到k2个点坐标【1，1600，20，3】
        src_knn_fine = src_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])  # [bs, num_seeds, k, 3]
        tgt_knn_fine = tgt_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])
        # 计算他们彼此之间的距离
        src_dist = ((src_knn_fine[:, :, :, None, :] - src_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn_fine[:, :, :, None, :] - tgt_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_measure = (cross_dist < self.d_thre * 2).float()
        local_SC2_measure = torch.matmul(local_hard_measure, local_hard_measure) / k2
        local_SC_measure = torch.clamp(1 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        # local_SC2_measure = local_SC_measure * local_SC2_measure
        local_SC2_measure = local_SC_measure
        local_SC2_measure = local_SC2_measure.view([-1, k2, k2])


        #################################
        # Power iteratation to get the inlier probability
        #################################
        # 对角线赋值0
        local_SC2_measure[:, torch.arange(local_SC2_measure.shape[1]), torch.arange(local_SC2_measure.shape[1])] = 0
        # 计算主导特征向量
        total_weight = self.cal_leading_eigenvector(local_SC2_measure, method='power')
        total_weight = total_weight.view([bs, -1, k2])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        # 通过并行每个子集的加权最小二乘法计算变换
        #################################
        total_weight = total_weight.view([-1, k2])
        # 1600组20个点坐标
        src_knn = src_knn_fine
        tgt_knn = tgt_knn_fine
        src_knn, tgt_knn = src_knn.view([-1, k2, 3]), tgt_knn.view([-1, k2, 3])

        #################################
        # compute the rigid transformation for each seed by the weighted SVD
        # 通过加权SVD计算每个种子的刚性变换矩阵
        #################################
        seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)
        seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])

        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        # 计算每个假设的内数（全部点*R+t），并找到每个点云对的最佳变换
        #################################通过字符串表示法执行各种张量操作全部点*R+t
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3,
                                                                    3:4]  # [bs, num_seeds, num_corr, 3]
        #################################
        # calculate the inlier number for each hypothesis, and find the best transformation for each point cloud pair
        #################################
        pred_position = pred_position.permute(0, 1, 3, 2)
        # 计算2范数【1，1600，8000】（预测坐标与实际坐标距离）
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
        # 距离小于阈值0.6的保留，具体有几个小于阈值
        seedwise_fitness = torch.sum((L2_dis < self.inlier_threshold).float(), dim=-1)  # [bs, num_seeds]
        # 得到最好的刚性矩阵的index，也就是最好的种子点集
        batch_best_guess = seedwise_fitness.argmax(dim=1)
        # 得到具体多少个点的值小于阈值
        best_guess_ratio = seedwise_fitness[0, batch_best_guess]
        # 找到最好的那个刚性矩阵
        final_trans = seedwise_trans.gather(dim=1,index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)

        return final_trans
    # 计算主特征向量（特征值最大对应的特征向量）
    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        使用幂迭代算法或 torch.symeig 计算主导特征向量
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix    correspond的相似度矩阵
            - method: select different method for calculating the learding eigenvector. 方法
        Output:
            - solution: [bs, num_corr] leading eigenvector  主征向量
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                # 用于检查两个张量是否在给定的 tolerance 范围内相等
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            # 用于计算对称矩阵的特征值和特征向量的函数
            e, v = torch.symeig(M, eigenvectors=True)
            # 取最大特征值对应的特征向量
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def cal_confidence(self, M, leading_eig, method='eig_value'):
        """
        Calculate the confidence of the spectral matching solution based on spectral analysis.
        Input:
            - M:          [bs, num_corr, num_corr] the compatibility matrix
            - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
        Output:
            - confidence
        """
        if method == 'eig_value':
            # max eigenvalue as the confidence (Rayleigh quotient)
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                        leading_eig[:, None, :] @ leading_eig[:, :, None])
            confidence = max_eig_value.squeeze(-1)
            return confidence
        elif method == 'eig_value_ratio':
            # max eigenvalue / second max eigenvalue as the confidence
            max_eig_value = (leading_eig[:, None, :] @ M @ leading_eig[:, :, None]) / (
                        leading_eig[:, None, :] @ leading_eig[:, :, None])
            # compute the second largest eigen-value
            B = M - max_eig_value * leading_eig[:, :, None] @ leading_eig[:, None, :]
            solution = torch.ones_like(B[:, :, 0:1])
            for i in range(self.num_iterations):
                solution = torch.bmm(B, solution)
                solution = solution / (torch.norm(solution, dim=1, keepdim=True) + 1e-6)
            solution = solution.squeeze(-1)
            second_eig = solution
            second_eig_value = (second_eig[:, None, :] @ B @ second_eig[:, :, None]) / (
                        second_eig[:, None, :] @ second_eig[:, :, None])
            confidence = max_eig_value / second_eig_value
            return confidence
        elif method == 'xMx':
            # max xMx as the confidence (x is the binary solution)
            # rank = torch.argsort(leading_eig, dim=1, descending=True)[:, 0:int(M.shape[1]*self.ratio)]
            # binary_sol = torch.zeros_like(leading_eig)
            # binary_sol[0, rank[0]] = 1
            confidence = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
            confidence = confidence.squeeze(-1) / M.shape[1]
            return confidence
    # 对刚性转换矩阵进行优化
    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, it_num, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        使用初始转换矩阵执行后优化（全部点），该矩阵仅在测试期间采用。
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        inlier_threshold = 1.2

        # inlier_threshold_list = [self.inlier_threshold] * it_num

        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * it_num#迭代次数
        else:  # for KITTI
            inlier_threshold_list = [1.2] * it_num

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            # src_keypts经过initial_trans得到的预测点 R*p+t
            warped_src_keypts = transform(src_keypts, initial_trans)
            # 计算预测点与实际点距离
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            # 计算有个距离小于阈值
            inlier_num = torch.sum(pred_inlier)
            # 内点个数前后差别不大，就跳出
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            #     用小于阈值的那群点再计算刚性转换矩阵
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans
    # 得到出匹配的点坐标，correspond=(src[i],tgt[i])
    def match_pair(self, src_keypts, tgt_keypts, src_features, tgt_features):
        N_src = src_features.shape[1]
        N_tgt = tgt_features.shape[1]
        # use all point or sample points.
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_desc = src_features[:, src_sel_ind, :]
        tgt_desc = tgt_features[:, tgt_sel_ind, :]
        src_keypts = src_keypts[:, src_sel_ind, :]
        tgt_keypts = tgt_keypts[:, tgt_sel_ind, :]

        # match points in feature space.算出src_desc与tgt_desc距离，distance.shape：【8000，8000】
        distance = torch.sqrt(2 - 2 * (src_desc[0] @ tgt_desc[0].T) + 1e-6)
        # 加一个维度【1，8000，8000】
        distance = distance.unsqueeze(0)
        # 找到distance【0】中距离最小的index（按行），即src_desc与哪个tgt_desc距离最近
        source_idx = torch.argmin(distance[0], dim=1)
        # corr = torch.cat([torch.arange(source_idx.shape[0])[:, None].cuda(), source_idx[:, None]], dim=-1)
        # 将source_idx的index与内部元素链接在一起，corr.shape：【8000，2】，里面的元素[[0,3733],[1,5678],...]
        # 代表最佳匹配【src_desc的index，tgt_desc的index】
        corr = torch.cat([torch.arange(source_idx.shape[0])[:, None], source_idx[:, None]], dim=-1)
        # generate correspondences,得到匹配好的点坐标，他们是一一对应关系，即src_keypts_corr【0】与tgt_keypts_corr【0】是最佳匹配
        src_keypts_corr = src_keypts[:, corr[:, 0]]
        tgt_keypts_corr = tgt_keypts[:, corr[:, 1]]

        return src_keypts_corr, tgt_keypts_corr
    # 得到调整后的刚性矩阵
    def SC2_PCR(self, src_keypts, tgt_keypts):
        """
        Input:输入的已经是匹配好的:src[i]匹配tgt[i]
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix.
            - pred_labels:  [bs, num_corr], the predicted inlier/outlier label (0,1), for classification loss calculation.
        """
        bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]

        #################################
        # downsample points
        #################################
        if num_corr > self.max_points:
            src_keypts = src_keypts[:, :self.max_points, :]
            tgt_keypts = tgt_keypts[:, :self.max_points, :]
            num_corr = self.max_points

        #################################
        # compute cross dist
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        # 算出每对correspond的距离差，cross_dist的shape为【1，8000，8000】
        cross_dist = torch.abs(src_dist - target_dist)

        #################################
        # compute first order measure
        #################################
        SC_dist_thre = self.d_thre
        # sc correspondences相似度矩阵【1，8000，8000】
        SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
        # 没有核函数直接去与阈值比较后的硬sc矩阵
        hard_SC_measure = (cross_dist < SC_dist_thre).float()
        # a=SC_measure==hard_SC_measure
        #################################
        # select reliable seed correspondences
        #################################
        confidence = self.cal_leading_eigenvector(SC_measure, method='power')
        seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=int(num_corr * self.ratio))

        #################################
        # compute second order measure计算sc二阶度量
        # cross_dist每对correspond的距离差，d_thre每对correspond的距离差阈值
        #################################
        SC2_dist_thre = self.d_thre / 2
        hard_SC_measure_tight = (cross_dist < SC2_dist_thre).float()
        # index:[1,1600,8000],找到种子与其他correspond硬sc相似度，并组成矩阵。hard_SC_measure与hard_SC_measure_tight
        # 不同在于组成矩阵时距离与之不同
        seed_hard_SC_measure = hard_SC_measure.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_corr))
        seed_hard_SC_measure_tight = hard_SC_measure_tight.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_corr))
        # seed_hard_SC_measure_tight:[1,1600,8000],hard_SC_measure_tight:[1,8000,8000]
        # ,seed_hard_SC_measure:[1,1600,8000],先矩阵乘，后元素乘
        # (seed,cor_i)*(cor_i,cor_k)对全部i循环，即可求出seed和几个cor_k有共同同类（因为同类为1，非为0），最后*seed_hard_SC_measure
        # 是为了将自己和自己相乘的数抑制掉，详细的见文章公式8
        SC2_measure = torch.matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure

        #################################
        # compute the seed-wise transformations and select the best one计算种子转换并选择最佳转换
        #################################最佳变换矩阵
        final_trans = self.cal_seed_trans(seeds, SC2_measure, src_keypts, tgt_keypts)

        #################################
        # refine the result by recomputing the transformation over the whole set通过重新计算整个集合的变换来优化结果
        #################################
        final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts, 20)

        return final_trans

    def estimator(self, src_keypts, tgt_keypts, src_features, tgt_features):
        """
        Input:
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
            - src_features: [bs, num_corr, C]
            - tgt_features: [bs, num_corr, C]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix
            - pred_trans:   [bs, num_corr], the predicted inlier/outlier label (0,1)
            - src_keypts_corr:  [bs, num_corr, 3], the source points in the matched correspondences
            - tgt_keypts_corr:  [bs, num_corr, 3], the target points in the matched correspondences
        """
        #################################
        # generate coarse correspondences
        ##################匹配好的点，correspond【i】：【src_keypts_corr【i】，tgt_keypts_corr【i】】
        src_keypts_corr, tgt_keypts_corr = self.match_pair(src_keypts, tgt_keypts, src_features, tgt_features)

        #################################计算刚性转换矩阵
        # use the proposed SC2-PCR to estimate the rigid transformation
        #################################
        pred_trans = self.SC2_PCR(src_keypts_corr, tgt_keypts_corr)
        # 得到预测点坐标
        frag1_warp = transform(src_keypts_corr, pred_trans)
        distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
        pred_labels = (distance < self.inlier_threshold).float()#根据变换矩阵、阈值得到所有target点的标签值，0，1，即内点还是野点

        return pred_trans, pred_labels, src_keypts_corr, tgt_keypts_corr
