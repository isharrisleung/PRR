import torch
import torch.nn as nn


# 超参
TIME_STEP = 28 # rnn 时序步长数
INPUT_SIZE = 28 # rnn 的输入维度
BATCH_SIZE = 512
HIDDEN_SIZE = 64 # of rnn 隐藏单元个数
EPOCHS=10 # 总共训练次数
LR = 0.01
h_state = None # 隐藏层状态
train_loss = []  # 误差汇总


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_SIZE, 10)  # (hidden_size, output_size)
    
    def forward(self, x):
        out_state, h_state = self.gru(x, None)

        outs = self.out(out_state[:, -1, :])
        # print(outs.shape)
        return outs

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征数
        self.out_features = out_features   # 节点表示向量的输出特征数
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # 初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[0]
        state_len = out_state.size()[0]
        attn_energies = torch.zeros(state_len, seq_len).to(device)
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.score(out_state[i], history[j])
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            # tanh
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


# ##############long###########################
class TrajPreAttnAvgLongUser(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        # rnn unit
        self.rnn = nn.GRU(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, history_loc, history_tim, history_count, uid, target_len):
        h1 = torch.zeros(1, 1, self.hidden_size)
        c1 = torch.zeros(1, 1, self.hidden_size)
        h1 = h1.to(device)
        c1 = c1.to(device)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        loc_emb_history = self.emb_loc(history_loc).squeeze(1)
        tim_emb_history = self.emb_tim(history_tim).squeeze(1)
        count = 0
        loc_emb_history2 = torch.zeros(len(history_count), loc_emb_history.size()[-1]).to(device)
        tim_emb_history2 = torch.zeros(len(history_count), tim_emb_history.size()[-1]).to(device)
        for i, c in enumerate(history_count):
            if c == 1:
                tmp = loc_emb_history[count].unsqueeze(0)
            else:
                tmp = torch.mean(loc_emb_history[count:count + c, :], dim=0, keepdim=True)
            loc_emb_history2[i, :] = tmp
            tim_emb_history2[i, :] = tim_emb_history[count, :].unsqueeze(0)
            count += c

        history = torch.cat((loc_emb_history2, tim_emb_history2), 1)
        history = F.tanh(self.fc_attn(history))

        out_state, h1 = self.rnn(x, h1)

        out_state = out_state.squeeze(1)
        # out_state = F.selu(out_state)

        attn_weights = self.attn(out_state[-target_len:], history).unsqueeze(0)
        context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0)
        out = torch.cat((out_state[-target_len:], context), 1)  # no need for fc_attn

        uid_emb = self.emb_uid(uid).repeat(target_len, 1)
        out = torch.cat((out, uid_emb), 1)
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y)

        return score


class TrajPreLocalAttnLong(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreLocalAttnLong, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
        self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim, target_len):
        h1 = torch.zeros(1, 1, self.hidden_size)
        h2 = torch.zeros(1, 1, self.hidden_size)
        c1 = torch.zeros(1, 1, self.hidden_size)
        c2 = torch.zeros(1, 1, self.hidden_size)
        h1 = h1.to(device)
        h2 = h2.to(device)
        c1 = c1.to(device)
        c2 = c2.to(device)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            hidden_history, h1 = self.rnn_encoder(x[:-target_len], h1)
            hidden_state, h2 = self.rnn_decoder(x[-target_len:], h2)
        elif self.rnn_type == 'LSTM':
            hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(x[-target_len:], (h2, c2))

        hidden_history = hidden_history.squeeze(1)
        hidden_state = hidden_state.squeeze(1)
        attn_weights = self.attn(hidden_state, hidden_history).unsqueeze(0)
        context = attn_weights.bmm(hidden_history.unsqueeze(0)).squeeze(0)
        out = torch.cat((hidden_state, context), 1)  # no need for fc_attn
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y)

        return score