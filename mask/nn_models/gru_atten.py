import torch
import torch.nn as nn

__all__ = ['AttenGRU']


class AttenGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_classes=2, n_layers=2):
        super(AttenGRU, self).__init__()
        self.feature_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = n_classes
        self.num_layers = n_layers
        self.gru = nn.GRU(self.feature_dim, self.hidden_dim, self.num_layers, dropout=0.2)
        self.fc = nn.Linear(self.hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        # atten parameter
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.weight_proj.data.uniform_(-0.1, 0.1)
        self.weight_W.data.uniform_(-0.1, 0.1)
        self.bias.data.zero_()

    def batch_soft_atten(self, seq, W, bias, v):
        s = []
        batch_size = seq.shape[1]
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], W)
            _s_bias = _s + bias.expand(bias_dim[0], batch_size).transpose(0, 1)
            _s_bias = torch.tanh(_s_bias)
            _s = torch.mm(_s_bias, v)
            s.append(_s)
        s = torch.cat(s, dim=1)
        soft = self.softmax(s)
        return soft

    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = []
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            attn_vectors.append(h_i)
        attn_vectors = torch.cat(attn_vectors, dim=0)
        return torch.sum(attn_vectors, 0)

    def soft_attention(self, ht):
        atten_alpha = self.batch_soft_atten(ht, self.weight_W, self.bias, self.weight_proj)
        atten_vects = self.attention_mul(ht, atten_alpha.transpose(0, 1))
        return atten_vects

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.transpose(0, 1)  # batch second
        self.gru.flatten_parameters()
        x, ht = self.gru(x)
        x = self.soft_attention(x)
        x = self.fc(x)
        return x, x


def test():
    data = torch.rand(128, 1, 99, 64)
    net = AttenGRU(64, n_classes=2)
    result, _ = net(data)
    print(result.shape)


if __name__ == "__main__":
    test()
