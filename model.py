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

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)

class AttnDecoderRNN(nn.Module):  #  'Neural Machine Translation by Jointly Learning to Align and Translate'
    def __init__(self, hidden_size, output_size, dropout_p=args.dropout, max_length=args.MAX_LENGTH): # 
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size) # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  # 
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self): # Initial hidden-layer
        return torch.zeros(1, 1, self.hidden_size, device=device)