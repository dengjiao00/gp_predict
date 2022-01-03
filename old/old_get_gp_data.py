import os
import numpy as np
import torch
from torch import nn, optim
from torch.serialization import save
from torch.utils.data.dataset import TensorDataset
from plot_imgs import plot_more
import tushare as ts
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

"""目标：通过前 windows_size - 1 天的收盘价来预测今天的收盘价
"""


def get_gp_data(window_size, input_size, test_size):
    # 获取股票数据
    pro = ts.pro_api(
        '2c641122001c8bd3411eeb1251849e4d54be01105ade1e90efab41c9')
    df = pro.daily(ts_code='000001.SZ', start_date='2015-01-01',
                   end_date='2020-02-25')
    # 获取每天的收盘价
    df = df.iloc[::-1]
    df.reset_index(inplace=True)
    dataset = np.array(df.loc[:, ['close']]).squeeze()
    # TODO: 可以改 test_size 来观察模型可视化的对比效果；通过训练集和数据集比例划分
    # 分割训练集和数据集（shuffle为False表示不打乱原始数据的顺序）
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, shuffle=False)

    # TODO: 按window_size分割训练集和数据集
    train_set = sliding_window(train_dataset, window_size=window_size)
    test_set = sliding_window(test_dataset, window_size=window_size)

    # 转成numpy数组
    # numpy 来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多（该结构也可以用来表示矩阵（matrix）），支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库 [1]  。
    train_set = np.array(train_set, dtype=np.float32).reshape(
        (-1, window_size, input_size))
    test_set = np.array(test_set, dtype=np.float32).reshape(
        (-1, window_size, input_size))
    print(f'Shape of train_set is {train_set.shape}')
    print(f'Shape of test_set is {test_set.shape}')
    return train_set, test_set


def sliding_window(seq, window_size):
    """根据窗口大小来划分数据
    数据的连续切割
    """
    result = []
    for i in range(len(seq) - window_size):
        result.append(seq[i:i + window_size])
    return result


"""

"""

# TODO ： RNN模型


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModule, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        o, h = self.rnn(x)
        o = self.linear(o)
        h = h.squeeze(0)
        return o, h

# TODO ： LSTM模型


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


def train_epoch(net, train_loader, optimizer, loss_func, device):
    # 进入训练模式
    net.train()

    loss_sum, num_samples = 0.0, 0
    # 从训练数据集加载器中获取一个 batch_size 的数据
    for (data, ) in train_loader:
        # data大小(shape) -> (batch_size, window_size, input_size)
        # 将数据搬到 device 上
        data = data.to(device)
        # data_x -> 特征  data_y -> 标签
        """
        例： window_size=4    
        data=[
            [[1], [2], [3], [4]],
            [[5], [6], [7], [8]]
        ]
        data_x, data_y = data[:, :-1], data[:, -1]
            其中 data_x = [
                [[1], [2], [3]],
                [[5], [6], [7]]
            ]
            data_y = [
                [4],
                [8]
            ]
        """
        data_x, data_y = data[:, :-1], data[:, -1]
        o, _ = net(data_x)
        y_hat = o[:, -1]

        # 计算loss
        loss = loss_func(data_y, y_hat)
        # 清空所有参数的梯度
        optimizer.zero_grad()
        # 通过 BP算法 反向传播（**计算**参数的梯度） loss
        loss.backward()
        # 通过参数的梯度来**更新参数值**（梯度下降）
        optimizer.step()

        # loss 求和
        loss_sum += loss.detach().cpu().numpy().tolist()
        # 取第一维的元素个数: batch_size
        num_samples += data.size(0)

    return loss_sum / num_samples


def test_epoch(net, test_loader, loss_func, device):
    # 进入测试模式
    net.eval()

    loss_sum, num_samples = 0.0, 0
    # TODO 评价指标：MAE，RMSE，MSE，MAPE
    # TODO MAE Score
    # TODO RMSE Score
    # 还可以加上MSE分数、MAPE分数
    mae_sum, rmse_sum = 0.0, 0.0
    # 不使用参数的梯度
    with torch.no_grad():
        for (data, ) in test_loader:
            data = data.to(device)
            data_x, data_y = data[:, :-1], data[:, -1]

            o, _ = net(data_x)
            y_hat = o[:, -1, :]

            loss = loss_func(data_y, y_hat)

            loss_sum += loss.detach().cpu().numpy().tolist()
            num_samples += data.size(0)

            mae_sum += mean_absolute_error(data_y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
            rmse_sum += np.sqrt(mae_sum)

    return loss_sum / num_samples, mae_sum / num_samples, rmse_sum / num_samples


def test_model(net, X, days, device):
    net.eval()
    y_hat_list = []
    with torch.no_grad():
        for day in range(days):
            X = X.to(device)
            o, _ = net(X)
            y_hat = o[:, -1, :][0]
            y_hat_list.append(y_hat.detach().cpu().numpy()[0])
            X = torch.cat([X[:, 1:], y_hat.unsqueeze(dim=0).unsqueeze(dim=0)], dim=1)

    return np.array(y_hat_list)


def main(window_size=4, input_size=1, output_size=1, hidden_size=256, batch_size=2, lr=0.0001, num_epoches=30, test_size=0.2, net_name='lstm', save_model=True, save_path=None):
    """[summary]

    Args:
        window_size ([type]): 窗口大小，可改，写报告
        input_size ([type]): 输入的特征（收盘价）数量
        output_size ([type]): 输出的特征（收盘价）数量
        hidden_size ([type]): 隐藏层大小，可改，写报告【128、256、1024】
        batch_size ([type]): 批量处理的个数（window_size的个数)，可改，写报告
        lr ([type]): 学习率，可改，写报告【不可太大，也不要太小，太小太慢，太大会左右震荡】
        num_epoches ([type]): 训练轮次，可改，写报告
    """

    # 调用函数，获取训练集和测试集
    train_set, test_set = get_gp_data(
        window_size=window_size, input_size=input_size, test_size=test_size)
    
    test_data_x = test_set[-1:,:window_size-1]
    test_set = test_set[:-1]
    # TODO： TensorDataset将numpy的数据转成tensor（张量，多维向量）
    train_dataset = TensorDataset(torch.from_numpy(train_set))
    test_dataset = TensorDataset(torch.from_numpy(test_set))

    # 构建数据集和测试集的加载器,drop_last是否丢弃最后的不足batch_size的数据，训练集需要打乱原始数据，测试集不用打乱
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 依据实际运行机器的情况选择在CPU还是GPU上运行
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    if net_name.lower() == 'rnn':
        net = RNNModule(input_size, hidden_size, output_size)
    elif net_name.lower() == 'lstm':
        net = LSTMModel(input_size, hidden_size, output_size)
    else:
        raise NameError('net_name 参数错误，只能为 rnn 或 lstm')

    # 将数据的模型放到对应选择的device上运行
    net.to(device)
    # 损失函数：度量预测值和真实值之间的差距
    loss_func = nn.MSELoss()

    # 优化器：优化参数
    # net.parameters() 网络的参数
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_loss_list, test_loss_list = [], []
    mae_list, rmse_list = [], []
    # 模型 训练/测试 num_epoches轮
    for epoch in range(1, num_epoches + 1):
        # 训练
        train_loss = train_epoch(
            net, train_loader, optimizer, loss_func, device)
        # 测试
        test_loss, mae_score, rmse_score = test_epoch(
            net, test_loader, loss_func, device)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        mae_list.append(mae_score)
        rmse_list.append(rmse_score)

        print(f'epoch: {epoch} train_loss: {train_loss:.4f} test_loss: {test_loss:.4f} mae_score: {mae_score:.4f} rmse_score: {rmse_score:.4f}')

    test_data_y_predict = test_model(net, torch.from_numpy(test_data_x), 5, device)
    plot_predict = np.concatenate([test_data_x.squeeze(0).squeeze(1), test_data_y_predict])

    if save_model and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # 保存训练好的模型
        torch.save(net, os.path.join(save_path, 'model.pkl'))
        # 保存实验结果，方便之后画图
        np.savez(os.path.join(save_path, f'result.npz'),
                 num_epoches=num_epoches,
                 train_loss_list=train_loss_list,
                 test_loss_list=test_loss_list,
                 mae_list=mae_list,
                 rmse_list=rmse_list)
    return {
        'net_name': net_name,
        'num_epoches': num_epoches,
        'train_loss_list': train_loss_list,
        'test_loss_list': test_loss_list,
        'mae_list': mae_list,
        'rmse_list': rmse_list,
        'plot_predict': plot_predict
    }


# TODO 程序入口
if __name__ == '__main__':
    result1 = main(window_size=51, input_size=1, output_size=1, hidden_size=256, batch_size=32, 
    lr=0.0001, num_epoches=30, test_size=0.2, net_name='lstm', save_model=True, save_path='a')
    result2 = main(window_size=51, input_size=1, output_size=1, hidden_size=256, batch_size=32, 
    lr=0.0005, num_epoches=50, test_size=0.2, net_name='rnn', save_model=True, save_path='a')
    plot_more([result2, ])
