from torch import nn


class RNNModel(nn.Module):
    """RNN 模型"""

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        o, h = self.rnn(x)
        o = self.linear(o)
        h = h.squeeze(0)
        return o, h


class LSTMModel(nn.Module):
    """单层的LSTM模型"""
    # 定义pytortch模型的组成单元

    def __init__(self, input_size, hidden_size, output_size):
        # 继承  nn.Module-->模型https://blog.csdn.net/qq_27825451/article/details/90550890
        super(LSTMModel, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 线性层
        self.linear = nn.Linear(hidden_size, output_size)

    # 实现模型需要实现forward方法，怎么将输入变成输出，公式的表达
    def forward(self, x):
        o, (h, c) = self.lstm(x)
        o = self.linear(o)
        h = h.squeeze(0)
        return o, (h, c)
