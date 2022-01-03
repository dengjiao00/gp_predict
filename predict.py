from datetime import datetime
from copy import deepcopy

import torch

from stock import download_stock_from_api
from prepare_data import get_prepared_stock_data, sliding_window
from train_test import train_test

from sklearn.preprocessing import MinMaxScaler

def predict(net, config, scaler, from_api=False, pred_days=5):
    stock_code = config['stock_code']
    start = config['start']
    end = config['end']
    used_len = config['unuse_len']
    window_size = config['window_size']
    output_size = config['output_size']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')

    if from_api:
        print('正在从 tushare api 获取数据')
        prepared_stock_data = download_stock_from_api(
            stock_code[1:], start, end, True)
    else:
        print('正在从 网易股票网站 获取数据')
        prepared_stock_data = get_prepared_stock_data(
            stock_code, start, end, True)
    
    prepared_stock_data = prepared_stock_data[-used_len:]
    print(prepared_stock_data.shape)
    prepared_stock_data = scaler.transform(prepared_stock_data)

    dataset = sliding_window(prepared_stock_data,
                             window_size=window_size + output_size)

    # torch.Size([32, 50, 1]) torch.Size([32, 5, 1]) torch.Size([32, 50, 5]) torch.Size([32, 5, 1])
    data_X = torch.from_numpy(dataset[:, :-output_size, :])
    data_Y = torch.from_numpy(dataset[:, -output_size:, :])
    data_X, data_Y = data_X.to(device), data_Y.to(device)
    # TODO: 错误版本（忘记归一化）
    # true_value = data_X.numpy()[-1, :, -1].tolist()
    # 将输入数据反归一化
    true_value = scaler.inverse_transform(data_X.numpy().reshape(-1, 1))[:, 0].tolist()
    pred_value = deepcopy(true_value)
    print(data_X.shape)
    print(data_Y.shape)
    print(true_value)

    for i in range(1, pred_days + 1):
        o, _ = net(data_X)
        pred_Y = o[:, -1:, :].reshape(data_Y.shape)

        print('反归一化数据')
        data_y = data_Y.detach().cpu().numpy().reshape(-1, 1)
        y_hat = pred_Y.detach().cpu().numpy().reshape(-1, 1)
        data_y = scaler.inverse_transform(data_y)
        y_hat = scaler.inverse_transform(y_hat)
        pred_value.append(y_hat.item())
        print(f'第 {i} 天  真实值：{data_y.item():.4f} 预测值：{y_hat.item():.4f}')

        # 更新输入和输出数据
        data_X = torch.cat([data_X[:, 1:], pred_Y], dim=1)
        data_Y = torch.cat([data_Y[:, 1:], pred_Y], dim=1)

    return true_value, pred_value

if __name__ == '__main__':
    config = {
        'stock_code': '0000001',
        'test_size': 0.2,
        'input_size': 1,
        'hidden_size': 256,
        'output_size': 1,
        'window_size': 50,
        'net_name': 'rnn',
        'lr': 0.001,
        'batch_size': 16,
        'num_epochs': 2,
        'unuse_len': 52,
        'start': datetime(year=2015, month=1, day=1),
        'end': datetime(year=2020, month=2, day=25)
    }
    scaler = MinMaxScaler()
    net, train_loss_stat, test_loss_stat, mae, mape, mae_stat, mape_stat = train_test(config=config, scaler=False,show_figure=False)

    config['batch_size'] = 1
    predict(net, config, scaler)
    # stock_code = '0000001'
    # start = datetime(year=2015, month=1, day=1)
    # end = datetime(year=2020, month=2, day=25)
    # reverse = True
    # from_api = True
    # used_len = 52
    # window_size = 50
    # output_size = 1
    # show_count = 3
    # pred_days = 5
    # pt_path = 'rnn.pkl'

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f'use device: {device}')

    # if from_api:
    #     print('正在从 tushare api 获取数据')
    #     prepared_stock_data = download_stock_from_api(
    #         stock_code[1:], start, end, reverse)
    # else:
    #     print('正在从 网易股票网站 获取数据')
    #     prepared_stock_data = get_prepared_stock_data(
    #         stock_code, start, end, reverse)

    # prepared_stock_data = prepared_stock_data[-used_len:]

    # dataset = sliding_window(prepared_stock_data,
    #                          window_size=window_size + output_size)

    # # torch.Size([32, 50, 1]) torch.Size([32, 5, 1]) torch.Size([32, 50, 5]) torch.Size([32, 5, 1])
    # data_X = torch.from_numpy(dataset[:, :-output_size, :])
    # data_Y = torch.from_numpy(dataset[:, -output_size:, :])
    # data_X, data_Y = data_X.to(device), data_Y.to(device)

    # net = torch.load(pt_path)
    # net.eval()
    # net.to(device)

    # for i in range(1, pred_days + 1):
    #     o, _ = net(data_X)
    #     pred_Y = o[:, -1:, :].reshape(data_Y.shape)

    #     print(f'第 {i} 天  真实值：{data_Y.item():.4f} 预测值：{pred_Y.item():.4f}')

    #     # 更新输入和输出数据
    #     data_X = torch.cat([data_X[:, 1:], pred_Y], dim=1)
    #     data_Y = torch.cat([data_Y[:, 1:], pred_Y], dim=1)
