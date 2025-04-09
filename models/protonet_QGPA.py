""" Prototypical Network 


"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from models.attention import SelfAttention, QGPA
from models.gmmn import GMMNnetwork


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class ProtoNetAlignQGPASR(nn.Module):
    def __init__(self, args):
        super(ProtoNetAlignQGPASR, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = 'cosine'
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.use_align = args.use_align
        self.use_linear_proj = args.use_linear_proj
        self.use_supervise_prototype = args.use_supervise_prototype
        if args.use_high_dgcnn:
            self.encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        else:
            self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        if self.use_linear_proj:# 语义projection
            self.conv_1 = nn.Sequential(nn.Conv1d(args.train_dim, args.train_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.train_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.use_transformer = args.use_transformer
        if self.use_transformer:
            self.transformer = QGPA()

    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat, _ = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat, xyz = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)
        # prototype learning
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat)
        #prototypes = [bg_prototype] + fg_prototypes

        updated_fg_prototypes = []
        for fg_proto in fg_prototypes:
            fg_proto_updated = self.update_prototype(query_feat, fg_proto)  # 使用查询集更新前景原型
            updated_fg_prototypes.append(fg_proto_updated)

        # 背景原型更新
        #bg_proto_updated = self.update_prototype(query_feat, bg_prototype)
        updated_prototypes = [bg_prototype] + updated_fg_prototypes
        prototypes = updated_prototypes
        #print('proto',prototypes.size())
        ###Self-Reconstruction
        #通过第III - C节我们可以得到更符合查询特征分布的精化原型，但是它们可能会丢失原来从支持度集合中学习到的关键类和语义信息
        #我们用 softmax 函数计算支持特征 与原型之间的余弦相似度
        #重建后的支持掩码 ˆ M c,k S 预计将与原始支持点云地面实况掩码 M c,k S 的信息保持一致。我们称这一过程为自重构（SR）
        self_regulize_loss = 0
        if self.use_supervise_prototype:
            self_regulize_loss = self.sup_regulize_Loss(prototypes, support_feat, fg_mask, bg_mask)

        ###QGPA  查询引导原型适应  query(q) support(key)  prototype(value)计算注意力机制
        if self.use_transformer:
            prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0).repeat(query_feat.shape[0], 1, 1)
            support_feat_ = support_feat.mean(1)
            prototypes_all_post = self.transformer(query_feat, support_feat_, prototypes_all)
            prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)#更新原型
            similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]#计算query和原型的相似性

            query_pred = torch.stack(similarity, dim=1)
            #print("query_pred", query_pred.size())
            loss = self.computeCrossEntropyLoss(query_pred, query_y)#query预测和真实的loss#proto更新后，计算query和label的损失
        else:
            similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
            query_pred = torch.stack(similarity, dim=1)

            loss = self.computeCrossEntropyLoss(query_pred, query_y)
        align_loss = 0

        ###???????
        ##proto更新后，
        # 其特征与查询原型进行对齐，并计算支持特征与对齐后原型的相似度。并计算支持预测与真实损失来约束对齐原型。
        if self.use_align:
            align_loss_epi = self.alignLoss_trans(query_feat, query_pred, support_feat, fg_mask, bg_mask)
            align_loss += align_loss_epi

        prototypes_all_post = prototypes_all_post.clone().detach()
        return query_pred, loss + align_loss + self_regulize_loss, prototypes_all_post

    def forward_test_semantic(self, support_x, support_y, query_x, query_y, embeddings=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """

        query_feat, xyz = self.getFeatures(query_x)

        # prototype learning
        if self.use_transformer:
            prototypes_all_post = embeddings
            prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for prototype in prototypes_new]
            query_pred = torch.stack(similarity, dim=1)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)

        return query_pred, loss

    # Self-Reconstruction
    def sup_regulize_Loss(self, prototype_supp, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype suppoort self alignment branch
        计算support 和 prototype 的相似度  与真实的损失
        Args:
            prototypes: embedding features for query images
                expect shape: N x C x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            prototypes = [prototype_supp[0], prototype_supp[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation 真实
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2, xyz = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            if self.use_linear_proj:#语义projection
                return self.conv_1(torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1)), xyz
            else:
                return torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1), xyz
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)    #求平均值
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def calculateSimilarity_trans(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
            #print("feat", feat.size())
            #print("prototype", prototype.size())
            #print("similarity", similarity.size())
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2)**2

        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)

    def alignLoss_trans(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
这段代码用于计算一个原型对齐分支的损失函数。具体步骤如下：

获取预测掩码：根据预测的分割分数确定每个像素点的类别，然后生成二进制掩码，区分前景和背景。
计算查询原型：通过加权求和和归一化得到查询图像的原型特征。
计算支持损失：对每个任务（way），跳过没有前景像素的任务。对于每个支持图像（shot），将其特征与查询原型进行对齐，并计算支持特征与对齐后原型的相似度。
生成支持标签：创建支持图像的地面真值标签，其中前景标记为1，背景标记为0。
计算损失：使用交叉熵损失函数计算预测与真实标签之间的差异，并对损失进行归一化。
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0)
                prototypes_all_post = self.transformer(img_fts, qry_fts.mean(0).unsqueeze(0), prototypes_all)
                prototypes_new = [prototypes_all_post[0, 0], prototypes_all_post[0, 1]]

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes_new]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss
    def update_prototype(self, query_feat, prototype):
        """
        使用查询集中相似度最高的点及其近邻特征来更新原型
        """
        # 计算查询集每个点与原型的相似度
        similarity = self.calculateSimilarity(query_feat, prototype, method=self.dist_method)

        # 找到相似度最高的 k 个中心点 (num_neighbors=3)
        _, topk_indices = torch.topk(similarity, 3, largest=True, dim=-1)  # [batch_size, num_center_neighbors]

        # 提取这些中心点的邻域特征 (num_neighbors=5)
        knn_features = self.get_knn_features(query_feat, topk_indices, num_neighbors=5)  # [batch_size, feat_dim, num_neighbors]

        # 对 knn_features 进行平均，得到一个与原型相同维度的特征向量 [batch_size, feat_dim]
        knn_mean = torch.mean(knn_features, dim=2)  # 对 num_neighbors 维度进行平均

        # 使用这些邻域特征更新原型
        updated_proto = torch.mean((prototype + knn_mean)/2  ,dim=0) # 更新原型，平均方式
        #print("updated_proto",updated_proto.size())
        return updated_proto

    def get_knn_features(self, query_feat, topk_indices, num_neighbors):
        """
        提取查询集中相似度最高的点及其邻域特征
        :param query_feat: 查询集特征 [batch_size, feat_dim, num_points]
        :param topk_indices: 中心点的索引 [batch_size, num_center_neighbors]
        :param num_neighbors: 每个中心点的邻域中提取的邻近点数量
        :return: 返回 [batch_size, feat_dim, num_center_neighbors * num_neighbors]
        """
        batch_size, feat_dim, num_points = query_feat.shape  # 获取查询集的维度信息
        num_center_neighbors = topk_indices.shape[1]  # 中心点的数量

        # 扩展 topk_indices 的维度以便用于 gather 索引，将其扩展为 [batch_size, num_center_neighbors, 1]
        topk_indices = topk_indices.unsqueeze(2).expand(batch_size, num_center_neighbors,
                                                        feat_dim)  # [batch_size, num_center_neighbors, feat_dim]

        # 提取中心点特征，结果形状为 [batch_size, feat_dim, num_center_neighbors]
        center_feats = torch.gather(query_feat, 2,
                                    topk_indices.permute(0, 2, 1))  # [batch_size, feat_dim, num_center_neighbors]

        # 初始化一个列表存储每个中心点的邻域特征
        knn_features = []

        # 对每个批次的中心点进行处理
        for i in range(num_center_neighbors):
            center_feat = center_feats[:, :, i]  # 提取第 i 个中心点的特征 [batch_size, feat_dim]
            #print('center_feat', center_feat.size())
            # 计算中心点与查询集中所有点的欧氏距离 [batch_size, num_points]
            distances = torch.norm(query_feat - center_feat.unsqueeze(2), dim=1)
            #print('distances', distances.size())
            # 获取每个中心点最近的 num_neighbors 个邻近点的索引 [batch_size, num_neighbors]
            _, knn_idx = torch.topk(distances, 5, largest=False)

            # 将 knn_idx 扩展以用于 gather，扩展为 [batch_size, feat_dim, num_neighbors]
            knn_idx = knn_idx.unsqueeze(1).expand(batch_size, feat_dim, 5)
            #print('knn_idx', knn_idx.size())
            # 提取邻近点特征，形状为 [batch_size, feat_dim, num_neighbors]
            knn_feat = torch.gather(query_feat, 2, knn_idx)
            #print(center_feat.unsqueeze(2).size())
            knn_feat=torch.cat((knn_feat,center_feat.unsqueeze(2)),dim=2)
            #print('knn_feat', knn_feat.size())
            # 将邻近点特征加入列表
            knn_features.append(knn_feat)

        # 将所有中心点的邻近点特征拼接在一起，形状为 [batch_size, feat_dim, num_center_neighbors * num_neighbors]
        knn_features = torch.cat(knn_features, dim=2)
        #print('knn_features',knn_features.size())
        return knn_features
