"""Self Attention Module


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)
        self.w = nn.Parameter(torch.ones(2))
    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)
        attn1 = F.softmax(attn, dim=-1)

        attn2 = self.relu(attn) ** 2

        #attn = self.dropout(F.softmax(attn, dim=-1))

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        #print("w1",w1)
        #print("w2", w2)
        attn = attn1 *0.7 + attn2 * 0.3
        attn=self.dropout(attn)
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)
'''
class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5


        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.mk = nn.Linear(in_channel, 64, bias=False)
        self.mv = nn.Linear(64, 64, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        #print("self.out_channel ",self.out_channel)#256
        #q = self.q_map(x)  # (batch_size, out_channel, num_points)
        #print("x", x.size())x torch.Size([2, 256, 2048])
        x=x.transpose(1,2)#[2, 2048,256}
        attk=self.mk(x)
        #attk =  self.dropout(F.softmax(attk, dim=-1))
        #print("attk", attk.size())
        attk=F.softmax(attk,dim=1)
        #print("attk", attk.size())
        attnorm=attk/(1e-9 + attk.sum(dim=1, keepdim=True))
        y=self.mv(attnorm)
        #print("y",y.size())#y torch.Size([2, 2048, 256])

        #attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        #attn = self.dropout(F.softmax(attn, dim=-1))
        #y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)
'''
class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5

        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1, 2)
'''
# self.transformer(query_feat, support_feat_, prototypes_all)
class QGPA(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(QGPA, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 512
        self.q_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, query, support, prototype):
        '''
        print("query",query.size())
        print("support",support.size())
        print("prototype",prototype.size())
        query
        torch.Size([3, 320, 2048])
        support
        torch.Size([3, 320, 2048])
        prototype
        torch.Size([3, 4, 320])
        '''

        batch, dim = query.shape[0], query.shape[1]
        way = support.shape[0] + 1#+背景
        residual = prototype
        q = self.q_map(query.transpose(1, 2))
        if len(support.shape) == 4:
            support = support.squeeze()
        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)
        k = self.k_map(support.transpose(1, 2))
        v = self.v_map(prototype)
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])
        #print("q", q.size())
        #("k", k.size())

        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)
        #print("attn1", attn.size())
        attn = attn.reshape(batch, way, dim, dim)
        #("attn2",attn.size())
        attn = F.softmax(attn, dim=-1)
        #print("attn3", attn.size())
        v = v.unsqueeze(2)
        output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)
        output = self.dropout(self.fc(output)).transpose(1, 2)
        output = self.layer_norm(output + residual)
        return output

class Dualalign(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super(Dualalign, self).__init__()
        self.in_channel = self.out_channel = 320
        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 2048
        self.q_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.v_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, support):
        '''
        query: torch.Size([3, 320, 2048])
        support: torch.Size([3, 320, 2048])
        prototype: torch.Size([3, 4, 320])
        '''
        batch, feat_dim, num_points = query.shape  # 获取批量大小、特征维度和点的数量
        way = support.shape[0] + 1  # 加背景类

        # Query, Key 和 Value 映射
        q = self.q_map(support.transpose(1, 2))  # 将 support 作为 query 进行映射 [batch, proj_dim, num_points]
        k = self.k_map(query.transpose(1, 2))  # 将 query 作为 key 进行映射 [batch, proj_dim, num_points]
        #print("q", q.size())
        #print("k", k.size())

        v = query  # query 直接作为 value 使用 [batch, feat_dim, num_points]
        v=self.v_map(query.transpose(1, 2))
        #print("v", v.size())
        # 计算注意力
        attn = torch.matmul(q.transpose(1, 2), k) / self.temperature  # [batch, num_points, num_points]
        #print("attn",attn.size())
        attn = F.softmax(attn, dim=-1)  # 在最后一个维度上做 softmax
        #print("attn2", attn.size())
        # 使用 attention 更新 query
        query_output = torch.matmul(attn, v.transpose(1, 2))  # [batch, num_points, feat_dim]
        #print("query_output", query_output.size())
        #query_output = query_output.transpose(1, 2)  # [batch, feat_dim, num_points]
        #print("query", query.size())
        #print("query_output + query", (query_output + query).size())
        up=(query_output + query).transpose(1, 2)
        #print("up", up.size())
        # 通过残差连接和 layer normalization 更新 query
        # query_output = self.dropout(self.fc(query_output))  # [batch, feat_dim, num_points]
        updated_query = self.layer_norm(up).transpose(1, 2)  # 残差连接

        return updated_query





'''class Dualalign(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(Dualalign, self).__init__()
        self.in_channel = self.out_channel = 320

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 512
        self.q_map = nn.Conv1d(2048, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(2048, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, query, support, prototype):
        
        # 获取 batch size 和 dim
        batch, dim = query.shape[0], query.shape[1]
        way = support.shape[0] + 1  # +背景

        residual = prototype  # 保存原始的 prototype 作为残差连接

        # 计算 query, support 和 prototype 的投影
        q = self.q_map(query.transpose(1, 2))
        if len(support.shape) == 4:
            support = support.squeeze()

        support = torch.cat([support.mean(0).unsqueeze(0), support], dim=0)  # 将背景加入 support
        k = self.k_map(support.transpose(1, 2))

        # 对 prototype 进行投影
        v = self.v_map(prototype)  # prototype 作为 value

        # 对 query 做 reshape 操作，以适应计算
        q = q.view(q.shape[1], q.shape[2] * q.shape[0])  # [proj_dim, batch_size * num_points]
        k = k.view(k.shape[1], k.shape[2] * k.shape[0])  # [proj_dim, batch_size * num_points]

        # 第一个注意力机制：更新 prototype
        attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)  # 计算注意力分数

        attn = attn.reshape(batch, way, dim, dim)  # 调整形状
        attn1 = attn.reshape(batch, way, dim)
        print("attn1", attn1.size())
        attn = F.softmax(attn, dim=-1)  # 归一化
        v = v.unsqueeze(2)  # 扩展维度以便进行矩阵乘法
        prototype_output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)  # 更新 prototype
        prototype_output = self.dropout(self.fc(prototype_output)).transpose(1, 2)  # 应用 dropout 和全连接层
        updated_prototype = self.layer_norm(prototype_output + residual)  # 加入残差连接并做 layer normalization
        print("attn", attn.size())
        # ------------------ 新增部分：更新 query ------------------
        # Value for query
        v_query = query  # 直接使用 query 作为 value
        print("v_query", v_query.size())
        v_query = v_query.unsqueeze(2)  # [batch, feat_dim, 1, 2048]
        print("v_query",v_query.size())
        # 使用相同的 attention 更新 query
        query_output = torch.matmul(attn, v_query).squeeze(-1).transpose(1, 2)
        query_output = self.dropout(self.fc(query_output)).transpose(1, 2)
        updated_query = self.layer_norm(query_output + query)
        # -------------
        return updated_prototype, updated_query
        '''