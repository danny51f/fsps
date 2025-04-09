import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = torch.mean(x, dim=2, keepdim=True)  # 点云平均池化
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool, _ = torch.max(x, dim=2, keepdim=True)  # 点云最大池化
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2)  # [batch_size, channels, 1]
        return scale

class FEM(nn.Module):
    def __init__(self, inplanes=320, channel_rate=2, reduction_ratio=4):
        super(FEM, self).__init__()


        self.in_channels = inplanes
        #print(self.in_channels)
        self.inter_channels =128  #inplanes // channel_rate
        #if self.inter_channels == 0:
           # self.inter_channels = 1
        #print(self.inter_channels)
        # 通用特征学习
        self.common_v = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积

        self.Trans_s = nn.Sequential(
            nn.Linear(self.inter_channels, self.in_channels),
            nn.BatchNorm1d(self.in_channels)
        )
        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)

        self.Trans_q = nn.Sequential(
            nn.Linear(self.inter_channels, self.in_channels),
            nn.BatchNorm1d(self.in_channels)
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)

        self.key = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积
        self.query = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积

        self.ChannelGate = ChannelGate(self.in_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

    def forward(self, q, s):

        print("q",q.size())
        print("s", s.size())

        # 特征学习
        v_q = self.common_v(q.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]
        v_s = self.common_v(s.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]

        k_x = self.key(s.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]
        q_x = self.query(q.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]

        A_s = torch.bmm(k_x, q_x.transpose(1, 2))  # [batch_size, num_points_s, num_points_q]
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.transpose(1, 2).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.bmm(attention_s, v_s)  # [batch_size, num_points_s, inter_channels]
        p_s = p_s.transpose(1, 2)  # [batch_size, inter_channels, num_points_s]

        # Adjusting the shape for BatchNorm1d
        p_s = p_s.contiguous().view(-1, self.inter_channels)  # [batch_size * num_points_s, inter_channels]
        p_s = self.Trans_s(p_s)  # 应用转换层
        p_s = p_s.view(q.shape[0], q.shape[2], self.in_channels).transpose(1,
                                                                             2)  # [batch_size, in_channels, num_points_s]

        E_s = self.ChannelGate(s) * p_s + s  # 结合原始特征和调整后的特征

        q_s = torch.bmm(attention_q, v_q)
        q_s = q_s.transpose(1, 2)  # 转置以适应 Linear 层输入

        # Adjusting the shape for BatchNorm1d
        q_s = q_s.contiguous().view(-1, self.inter_channels)  # [batch_size * num_points_q, inter_channels]
        q_s = self.Trans_q(q_s)  # 应用转换层
        q_s = q_s.view(q.shape[0], q.shape[2], self.in_channels).transpose(1,
                                                                             2)  # [batch_size, in_channels, num_points_q]

        E_q = self.ChannelGate(q) * q_s + q  # 结合原始特征和调整后的特征
        print("E_q", E_q.size())
        print("E_s", E_s.size())
        return E_q, E_s

class FEM_5(nn.Module):
        def __init__(self, inplanes=320, channel_rate=2, reduction_ratio=4):
            super(FEM_5, self).__init__()

            self.in_channels = inplanes
            # print(self.in_channels)
            self.inter_channels = 128  # inplanes // channel_rate
            # if self.inter_channels == 0:
            # self.inter_channels = 1
            # print(self.inter_channels)
            # 通用特征学习
            self.common_v = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积

            self.Trans_s = nn.Sequential(
                nn.Linear(self.inter_channels, self.in_channels),
                nn.BatchNorm1d(self.in_channels)
            )
            nn.init.constant_(self.Trans_s[1].weight, 0)
            nn.init.constant_(self.Trans_s[1].bias, 0)

            self.Trans_q = nn.Sequential(
                nn.Linear(self.inter_channels, self.in_channels),
                nn.BatchNorm1d(self.in_channels)
            )
            nn.init.constant_(self.Trans_q[1].weight, 0)
            nn.init.constant_(self.Trans_q[1].bias, 0)

            self.key = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积
            self.query = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积

            self.ChannelGate = ChannelGate(self.in_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        def forward(self, q, s):
            #print("q", q.size())  # q: torch.Size([2, 320, 2048])
            #print("s", s.size())  # s: torch.Size([10, 320, 2048])

            # Reshape support (s) to separate batch_size and shot * n_way

            sf = s
            sf1 = torch.stack((s[0], s[1], s[2], s[3], s[4]),
                              dim=0)
            sf1 = torch.mean(sf1, dim=0, keepdim=True)
            sf2 = torch.stack((s[5], s[6], s[7], s[8], s[9]),
                              dim=0)
            sf2 = torch.mean(sf2, dim=0, keepdim=True)
            sf = torch.cat((sf1, sf2), dim=0)

            v_q = self.common_v(q.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]
            v_s = self.common_v(sf.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]

            k_x = self.key(sf.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]
            q_x = self.query(q.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]

            A_s = torch.bmm(k_x, q_x.transpose(1, 2))  # [batch_size, num_points_s, num_points_q]
            attention_s = F.softmax(A_s, dim=-1)

            A_q = A_s.transpose(1, 2).contiguous()
            attention_q = F.softmax(A_q, dim=-1)

            p_s = torch.bmm(attention_s, v_s)  # [batch_size, num_points_s, inter_channels]
            p_s = p_s.transpose(1, 2)  # [batch_size, inter_channels, num_points_s]

            # Adjusting the shape for BatchNorm1d
            p_s = p_s.contiguous().view(-1, self.inter_channels)  # [batch_size * num_points_s, inter_channels]
            p_s = self.Trans_s(p_s)  # 应用转换层
            p_s = p_s.view(q.shape[0], q.shape[2], self.in_channels).transpose(1,
                                                                               2)  # [batch_size, in_channels, num_points_s]
            #print("self.ChannelGate(sf) ",self.ChannelGate(sf).size())
            #print("p_s ", p_s.size())
            qqq=self.ChannelGate(sf) * p_s
            #print("qqq ", qqq.size())#qqq  torch.Size([2, 320, 2048])
            #print("s0 ", s[0].size())#s0  torch.Size([320, 2048])
            qqq=torch.mean(qqq, dim=0, keepdim=True)
            #print("qqq ", qqq.size())
            support_feat=s
            support_feat[0] = support_feat[0] + qqq#.repeat(192, 1)
            support_feat[1] = support_feat[1] + qqq#.repeat(192, 1)
            support_feat[2] = support_feat[2] + qqq#.repeat(192, 1)
            support_feat[3] = support_feat[3] + qqq#.repeat(192, 1)
            support_feat[4] = support_feat[4] + qqq#.repeat(192, 1)
            support_feat[5] = support_feat[5] + qqq#.repeat(192, 1)
            support_feat[6] = support_feat[6] + qqq#.repeat(192, 1)
            support_feat[7] = support_feat[7] + qqq#.repeat(192, 1)
            support_feat[8] = support_feat[8] + qqq#.repeat(192, 1)
            support_feat[9] = support_feat[9] + qqq#.repeat(192, 1)
            E_s=support_feat
            #E_s = self.ChannelGate(sf) * p_s + sf  # 结合原始特征和调整后的特征

            q_s = torch.bmm(attention_q, v_q)
            q_s = q_s.transpose(1, 2)  # 转置以适应 Linear 层输入

            # Adjusting the shape for BatchNorm1d
            q_s = q_s.contiguous().view(-1, self.inter_channels)  # [batch_size * num_points_q, inter_channels]
            q_s = self.Trans_q(q_s)  # 应用转换层
            q_s = q_s.view(q.shape[0], q.shape[2], self.in_channels).transpose(1,
                                                                               2)  # [batch_size, in_channels, num_points_q]

            E_q = self.ChannelGate(q) * q_s + q  # 结合原始特征和调整后的特征
            #print("E_q", E_q.size())
            #print("E_s", E_s.size())
            return E_q, E_s


class FEM_15(nn.Module):
    def __init__(self, inplanes=320, channel_rate=2, reduction_ratio=4):
        super(FEM_15, self).__init__()

        self.in_channels = inplanes
        # print(self.in_channels)
        self.inter_channels = 128  # inplanes // channel_rate
        # if self.inter_channels == 0:
        # self.inter_channels = 1
        # print(self.inter_channels)
        # 通用特征学习
        self.common_v = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积

        self.Trans_s = nn.Sequential(
            nn.Linear(self.inter_channels, self.in_channels),
            nn.BatchNorm1d(self.in_channels)
        )
        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)

        self.Trans_q = nn.Sequential(
            nn.Linear(self.inter_channels, self.in_channels),
            nn.BatchNorm1d(self.in_channels)
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)

        self.key = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积
        self.query = nn.Linear(self.in_channels, self.inter_channels)  # 使用线性层替代卷积

        self.ChannelGate = ChannelGate(self.in_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

    def forward(self, q, s):
        # print("q", q.size())  # q: torch.Size([2, 320, 2048])
        # print("s", s.size())  # s: torch.Size([10, 320, 2048])

        # Reshape support (s) to separate batch_size and shot * n_way

        sf = s
        sf1 = torch.stack((s[0], s[1], s[2], s[3], s[4]), dim=0)
        sf1 = torch.mean(sf1, dim=0, keepdim=True)
        sf2 = torch.stack((s[5], s[6], s[7], s[8], s[9]), dim=0)
        sf2 = torch.mean(sf2, dim=0, keepdim=True)
        sf3 = torch.stack((s[10], s[11], s[12], s[13], s[14]),
                          dim=0)
        sf3 = torch.mean(sf3, dim=0, keepdim=True)
        sf = torch.cat((sf1, sf2, sf3), dim=0)

        v_q = self.common_v(q.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]
        v_s = self.common_v(sf.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]

        k_x = self.key(sf.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]
        q_x = self.query(q.transpose(1, 2)).transpose(1, 2)  # [batch_size, num_points, inter_channels]

        A_s = torch.bmm(k_x, q_x.transpose(1, 2))  # [batch_size, num_points_s, num_points_q]
        attention_s = F.softmax(A_s, dim=-1)

        A_q = A_s.transpose(1, 2).contiguous()
        attention_q = F.softmax(A_q, dim=-1)

        p_s = torch.bmm(attention_s, v_s)  # [batch_size, num_points_s, inter_channels]
        p_s = p_s.transpose(1, 2)  # [batch_size, inter_channels, num_points_s]

        # Adjusting the shape for BatchNorm1d
        p_s = p_s.contiguous().view(-1, self.inter_channels)  # [batch_size * num_points_s, inter_channels]
        p_s = self.Trans_s(p_s)  # 应用转换层
        p_s = p_s.view(q.shape[0], q.shape[2], self.in_channels).transpose(1,
                                                                           2)  # [batch_size, in_channels, num_points_s]
        # print("self.ChannelGate(sf) ",self.ChannelGate(sf).size())
        # print("p_s ", p_s.size())
        qqq = self.ChannelGate(sf) * p_s
        # print("qqq ", qqq.size())#qqq  torch.Size([2, 320, 2048])
        # print("s0 ", s[0].size())#s0  torch.Size([320, 2048])
        qqq = torch.mean(qqq, dim=0, keepdim=True)
        # print("qqq ", qqq.size())
        support_feat = s
        support_feat[0] = support_feat[0] + qqq  # .repeat(192, 1)
        support_feat[1] = support_feat[1] + qqq  # .repeat(192, 1)
        support_feat[2] = support_feat[2] + qqq  # .repeat(192, 1)
        support_feat[3] = support_feat[3] + qqq  # .repeat(192, 1)
        support_feat[4] = support_feat[4] + qqq  # .repeat(192, 1)
        support_feat[5] = support_feat[5] + qqq  # .repeat(192, 1)
        support_feat[6] = support_feat[6] + qqq  # .repeat(192, 1)
        support_feat[7] = support_feat[7] + qqq  # .repeat(192, 1)
        support_feat[8] = support_feat[8] + qqq  # .repeat(192, 1)
        support_feat[9] = support_feat[9] + qqq  # .repeat(192, 1)
        support_feat[10] = support_feat[10] + qqq  # .repeat(192, 1)
        support_feat[11] = support_feat[11] + qqq  # .repeat(192, 1)
        support_feat[12] = support_feat[12] + qqq  # .repeat(192, 1)
        support_feat[13] = support_feat[13] + qqq  # .repeat(192, 1)
        support_feat[14] = support_feat[14] + qqq  # .repeat(192, 1)
        E_s = support_feat
        # E_s = self.ChannelGate(sf) * p_s + sf  # 结合原始特征和调整后的特征

        q_s = torch.bmm(attention_q, v_q)
        q_s = q_s.transpose(1, 2)  # 转置以适应 Linear 层输入

        # Adjusting the shape for BatchNorm1d
        q_s = q_s.contiguous().view(-1, self.inter_channels)  # [batch_size * num_points_q, inter_channels]
        q_s = self.Trans_q(q_s)  # 应用转换层
        q_s = q_s.view(q.shape[0], q.shape[2], self.in_channels).transpose(1,
                                                                           2)  # [batch_size, in_channels, num_points_q]

        E_q = self.ChannelGate(q) * q_s + q  # 结合原始特征和调整后的特征
        # print("E_q", E_q.size())
        # print("E_s", E_s.size())
        return E_q, E_s