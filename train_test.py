from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt

from prepare_data import get_dataset
from model import RNNModel, LSTMModel
from dataset import StockDataset


def train_test(from_api=False, scaler=None, config=None, show_figure=False):
    if config is None:
        raise RuntimeError('config 不能为空')
    stock_code = config['stock_code']
    start = config['start']
    end = config['end']
    test_size = config['test_size']
    input_size, hidden_size, output_size = config['input_size'], config['hidden_size'], config['output_size']
    # 使用的窗口大小
    window_size = config['window_size']
    net_name = config['net_name']
    lr = config['lr']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    unuse_len = config['unuse_len']

    train_X, train_Y, test_X, test_Y = get_dataset(
        stock_code, start, end, reverse=True, test_size=test_size, window_size=window_size,
        output_size=output_size, from_api=from_api, scaler=scaler, unuse_len=unuse_len)

    # 转成 tensor 格式
    train_X, train_Y, test_X, test_Y = torch.from_numpy(
        train_X), torch.from_numpy(train_Y), torch.from_numpy(test_X), torch.from_numpy(test_Y)
    train_dataset = StockDataset(
        train_X, train_Y)
    test_dataset = StockDataset(test_X, test_Y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')

    if net_name.lower() == 'rnn':
        net = RNNModel(1, hidden_size, output_size)
    elif net_name.lower() == 'lstm':
        net = LSTMModel(1, hidden_size, output_size)
    else:
        raise NameError('net_name 参数错误，只能为 rnn 或 lstm')
    net.to(device)

    # 损失函数
    loss_func = nn.MSELoss()
    # 参数优化器
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_loss_stat, test_loss_stat = [], []
    mae_stat, mape_stat, mse_stat = [], [], []
    logged = False
    for epoch in range(1, num_epochs + 1):
        num_samples = 0
        train_loss_sum = 0.0
        for data_x, data_y in train_loader:
            data_x, data_y = data_x.to(device), data_y.to(device)
            o, _ = net(data_x)
            y_hat = o[:, -1:, :].reshape(data_y.shape)
            # print(data_x.shape, data_y.shape, o.shape, y_hat.shape)

            # 计算loss
            loss = loss_func(data_y, y_hat)
            # 清空所有参数的梯度
            optimizer.zero_grad()
            # 通过 BP算法 反向传播（**计算**参数的梯度） loss
            loss.backward()
            # 通过参数的梯度来**更新参数值**（梯度下降）
            optimizer.step()

            train_loss_sum += loss.detach().cpu().numpy().tolist()
            num_samples += data_y.size(0)

        train_loss_epoch = train_loss_sum / num_samples
        train_loss_stat.append(train_loss_epoch)

        num_samples_test = 0
        test_loss_sum = 0
        mae_sum, mape_sum = 0, 0
        with torch.no_grad():
            for data_x, data_y in test_loader:
                data_x, data_y = data_x.to(device), data_y.to(device)
                o, _ = net(data_x)
                y_hat = o[:, -1:, :].reshape(data_y.shape)

                loss = loss_func(data_y, y_hat)

                test_loss_sum += loss.detach().cpu().numpy().tolist()
                num_samples_test += data_y.size(0)

                data_y = data_y.detach().cpu().numpy().reshape(data_y.shape[0], -1)
                y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[0], -1)

                if scaler is not None:
                    if not logged:
                        print('正在反归一化数据')
                        logged = True
                    data_y = scaler.inverse_transform(data_y)
                    y_hat = scaler.inverse_transform(y_hat)
                mae_sum += mean_absolute_error(data_y, y_hat).tolist()
                mape_sum += mean_absolute_percentage_error(data_y, y_hat).tolist()

        test_loss_epoch = test_loss_sum / num_samples_test
        test_loss_stat.append(train_loss_epoch)
        mae_stat.append(mae_sum / num_samples_test)
        mape_stat.append(mape_sum / num_samples)
        print(
            f'Epoch: {epoch} Train Loss: {train_loss_epoch:.4f} Test Loss: {test_loss_epoch:.4f}')

    num_samples = 0
    mae_sum = mape_sum = 0
    with torch.no_grad():
        net.eval()
        for data_x, data_y in test_loader:
            data_x, data_y = data_x.to(device), data_y.to(device)
            o, _ = net(data_x)
            y_hat = o[:, -1:, :].reshape(data_y.shape)

            num_samples += data_y.size(0)

            data_y = data_y.detach().cpu().numpy().reshape(data_y.shape[0], -1)
            y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[0], -1)

            if scaler is not None:
                if not logged:
                    print('正在反归一化数据')
                    logged = True
                data_y = scaler.inverse_transform(data_y)
                y_hat = scaler.inverse_transform(y_hat)
            mae_sum += mean_absolute_error(data_y, y_hat)
            mape_sum += mean_absolute_percentage_error(data_y, y_hat)

    print(
        f'On Test Dataset, MAE: {mae_sum / num_samples : .4f} MAPE: {mape_sum / num_samples: .4f}')

    torch.save(net,
               f'{net_name}_{input_size}_{hidden_size}_{output_size}_{lr}_{batch_size}_{num_epochs}.pkl')
    torch.save(net, f'{net_name}.pkl')

    if show_figure:
        plt.plot(range(1, num_epochs + 1), train_loss_stat, label='train loss')
        plt.plot(range(1, num_epochs + 1), test_loss_stat, label='test loss')
        plt.title('Loss')
        plt.show()
    
    return net, train_loss_stat, test_loss_stat, mae_sum / num_samples, mape_sum / num_samples, mae_stat, mape_stat


if __name__ == "__main__":
    # test_size = 0.2
    # input_size, hidden_size, output_size = 1, 256, 1
    # # 使用的窗口大小
    # window_size = 50
    # net_name = 'rnn'
    # lr = 0.001
    # batch_size = 16
    # num_epochs = 50
    # unuse_len = 60
    train_test(config={
        'stock_code': '0000001',
        'test_size': 0.2,
        'input_size': 1,
        'hidden_size': 256,
        'output_size': 1,
        'window_size': 50,
        'net_name': 'rnn',
        'lr': 0.001,
        'batch_size': 16,
        'num_epochs': 50,
        'unuse_len': 60,
    }, use_scaler=False,show_figure=True)
