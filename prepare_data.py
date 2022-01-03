from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stock import download_stock, download_stock_from_api


def get_prepared_stock_data(stock_code, start, end, reverse=False):
    stock_data = download_stock(stock_code, start, end)
    bio = BytesIO(stock_data)
    df = pd.read_csv(bio, encoding='gb2312')
    bio.close()

    # 将所有的数据按时间顺序从小到大排列
    if reverse:
        df = df.iloc[::-1]
        df.reset_index(inplace=True)
    return df.loc[:, ['收盘价', ]].values


def sliding_window(seq, window_size):
    """根据窗口大小来划分数据
    数据的连续切割
    """
    result = []
    for i in range(len(seq) - window_size):
        result.append(seq[i:i + window_size])

    return np.array(result, dtype=np.float32)


def get_dataset(stock_code, start, end, window_size, output_size, reverse=True, test_size=0.2, from_api=False, scaler=None, unuse_len=55):
    """获取实验数据集

    Args:
        stock_code ([str]): 股票代码
        start ([str or datetime]): 开始时间
        end ([str or datetime]): 结束时间
        reverse (bool, optional): 是否按时间从小到大排序. Defaults to True.
        test_size (float, optional): 测试集所占比例. Defaults to 0.2.
        window_size (int, optional): 窗口大小. Defaults to 2.

    Returns:
        [type]: [description]
    """
    # 获取股票数据
    if from_api:
        print('正在从 tushare api 获取数据')
        prepared_stock_data = download_stock_from_api(
            stock_code[1:], start, end, reverse)
    else:
        print('正在从 网易股票网站 获取数据')
        prepared_stock_data = get_prepared_stock_data(
            stock_code, start, end, reverse)

    print(prepared_stock_data.shape)
    # 为了方便使用 unuse_len 中的数据进行预测
    prepared_stock_data = prepared_stock_data[:-unuse_len]

    # 可以改 test_size 来观察模型可视化的对比效果；通过训练集和数据集比例划分
    # 分割训练集和数据集（shuffle为False表示不打乱原始数据的顺序）
    train_set, test_set = train_test_split(
        prepared_stock_data, test_size=test_size, shuffle=False)

    if scaler is not None:
        print(train_set.shape)
        print(f'正在归一化数据')
        train_set = scaler.fit_transform(train_set)
        test_set = scaler.transform(test_set)

    # 按window_size分割训练集和数据集
    train_set = sliding_window(
        train_set, window_size=window_size + output_size)
    test_set = sliding_window(test_set, window_size=window_size + output_size)

    # 转成numpy数组
    # numpy 来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多（该结构也可以用来表示矩阵（matrix）），支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库 [1]  。
    train_X = train_set[:, :-output_size, :]
    train_Y = train_set[:, -output_size:, :]
    test_X = test_set[:, :-output_size, :]
    test_Y = test_set[:, -output_size:, :]

    print(f'Shape of train_X is {train_X.shape}, train_Y is {train_Y.shape}')
    print(f'Shape of test_X is {test_X.shape}, test_Y is {test_Y.shape}')
    return train_X, train_Y, test_X, test_Y


if __name__ == "__main__":
    get_dataset(
        '0000001', '19901219', '20211118', reverse=True)
