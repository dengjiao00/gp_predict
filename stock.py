from datetime import datetime
from io import BytesIO

import tushare as ts
import requests
import progressbar
import pandas as pd


def download_stock_from_api(code, start, end, reverse=True):
    format_str = "%Y-%m-%d"
    if isinstance(start, datetime):
        start = start.strftime(format_str)
    if isinstance(end, datetime):
        end = end.strftime(format_str)
    pro = ts.pro_api(
        '2c641122001c8bd3411eeb1251849e4d54be01105ade1e90efab41c9')
    df = pro.daily(ts_code=f'{code}.SZ', start_date=start,
                   end_date=end)
    # 获取每天的收盘价
    if reverse:
        df = df.iloc[::-1]
        df.reset_index(inplace=True)
    return df.loc[:, ['close']].values


def download_stock(code, start, end, fields=None, save_path=None):
    """下载股票数据
    """
    if len(code) != 6:
        raise Exception("股票代码错误，请使用 6 位股票代码")
    code = f'1{code}'
    format_str = "%Y%m%d"
    if isinstance(start, datetime):
        start = start.strftime(format_str)
    if isinstance(end, datetime):
        end = end.strftime(format_str)

    # 默认只使用收盘价
    if fields is None:
        fields = ['TCLOSE', ]

    fields = ','.join(fields)

    url = f'http://quotes.money.163.com/service/chddata.html?code={code}&start={start}&end={end}&fields={fields}'

    if save_path is None:
        dst = BytesIO()
    else:
        dst = open(save_path, 'wb')

    print(f'下载链接: {url}')
    print(f'正在下载 {code} 股票数据...')
    resp = requests.get(url, stream=True)

    # 由于后端没有返回 Content-Length 所以手动设定一个值：5M
    max_length = 5 * 1024
    widgets = [
        'Process: ', progressbar.Percentage(), ' ', progressbar.Bar(
            marker='#', left='[', right=']'), ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()
    ]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=max_length).start()
    for chunk in resp.iter_content(chunk_size=64):
        if chunk:
            dst.write(chunk)
            dst.flush()
        pbar.update(len(chunk) + 1)
    pbar.finish()

    ret = None
    if save_path is None:
        dst.seek(0)
        ret = dst.getvalue()

    if dst is not None:
        dst.close()

    return ret

if __name__ == "__main__":
    stock_data = download_stock('0000001', '19901219', '20211118')
    df = pd.read_csv(BytesIO(stock_data), encoding='gb2312')
    print(df.head(5))
    print(df.shape)
    print(df.iloc[:, 3])
    print(df.iloc[:, 3].shape)
