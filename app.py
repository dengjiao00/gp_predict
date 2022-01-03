from datetime import datetime
from flask import Flask, json, request, render_template, jsonify
from train_test import train_test
from predict import predict
import pymysql
from sklearn.preprocessing import MinMaxScaler


conn = pymysql.connect(host='jzw.ink', port=3306,
                       user='root', passwd='123456', db='bb_gp_predict')


def make_res(train_loss_stat, test_loss_stat, mape_stat, mae_stat, true_value, pred_value):
    return {
        'predict': {
            'title': {
                'text': '根据50天预测5天的股票价格'
            },
            'tooltip': {
                'trigger': 'axis'
            },
            'legend': {
                'data': ['真实值', '预测值']
            },
            'xAxis': {
                'type': 'category',
                'data': list(range(1, len(pred_value) + 1))
            },
            'yAxis': {
                'type': 'value'
            },
            'series': [{
                'name': '真实值',
                'type': 'line',
                        'smooth': True,
                        'data': true_value
            }, {
                'name': '预测值',
                        'type': 'line',
                        'smooth': True,
                        'data': pred_value
            }]
        },
        'train_test_loss': {
            'title': {
                'text': '训练集和测试集上的Loss变化曲线'
            },
            'tooltip': {
                'trigger': 'axis'
            },
            'legend': {
                'data': ['Train Loss', 'Test Loss']
            },
            'xAxis': {
                'type': 'category',
                'data': list(range(1, len(train_loss_stat) + 1))
            },
            'yAxis': {
                'type': 'value'
            },
            'series': [{
                'name': 'Train Loss',
                'type': 'line',
                        'stack': 'Total',
                        'smooth': True,
                        'data': train_loss_stat
            }, {
                'name': 'Test Loss',
                        'type': 'line',
                        'stack': 'Total',
                        'smooth': True,
                        'data': test_loss_stat
            }]
        },
        'stat': {
            'title': {
                'text': 'mae'
            },
            'tooltip': {
                'trigger': 'axis'
            },
            'legend': {
                'data': ['mae', 'mape']
            },
            'xAxis': {
                'type': 'category',
                        'data': list(range(1, len(mape_stat) + 1))
            },
            'yAxis': {
                'type': 'value'
            },
            'series': [{
                'name': 'mae',
                        'type': 'line',
                        'stack': 'Total',
                        'smooth': True,
                        'data': mae_stat
            }, {
                'name': 'mape',
                        'type': 'line',
                        'stack': 'Total',
                        'smooth': True,
                        'data': mape_stat
            }]
        }
    }


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        sql = "SELECT DISTINCT `gp_num` FROM `raw`;"
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        gp_nums = sorted([r[0] for r in res])
        print(gp_nums)
        conn.commit()
        cursor.close()
        return render_template("index.html", gp_nums=gp_nums)
    elif request.method == "POST":
        gp_num = request.form['gp_num']
        model = request.form['model']

        sql = "SELECT `result` FROM `raw` where gp_num = %s;"
        cursor = conn.cursor()
        cursor.execute(sql, args=[gp_num, ])
        res = cursor.fetchone()
        print(res)
        result = res[0]
        cursor.close()
        if result is not None:
            result_json = json.loads(result)
        else:
            result_json = {}

        if model in result_json:
            print('从服务器获取结果')
            # train_loss_stat, test_loss_stat, mape_stat, mae_stat
            json_res = {'code': 0}
            model_result = result_json[model]
            json_res.update(**make_res(model_result['train_loss_stat'], model_result['test_loss_stat'],
                                       model_result['mape_stat'], model_result['mae_stat'],
                                       model_result['true_value'], model_result['pred_value']))
            return jsonify(json_res)
        else:
            config = {
                'stock_code': gp_num,
                'test_size': 0.2,
                'input_size': 1,
                'hidden_size': 256,
                'output_size': 1,
                'window_size': 50,
                'net_name': model,
                'lr': 0.0001,
                'batch_size': 16,
                'num_epochs': 25,
                'unuse_len': 52,
                'start': datetime(year=2018, month=3, day=5),
                'end': datetime(year=2020, month=2, day=25)
            }
            try:
                scaler = MinMaxScaler()
                net, train_loss_stat, test_loss_stat, mae, mape, mae_stat, mape_stat = train_test(
                    config=config,scaler=scaler, show_figure=False)
                config['batch_size'] = 1
                true_value, pred_value = predict(net= net, config=config,scaler=scaler)
                result_json.update({
                    f'{model}': {
                        'train_loss_stat': train_loss_stat,
                        'test_loss_stat': test_loss_stat,
                        'mae_stat': mae_stat,
                        'mape_stat': mape_stat,
                        'true_value': true_value,
                        'pred_value': pred_value,
                        'mae': mae,
                        'mape': mape
                    }
                })
                sql = "UPDATE `raw` SET `result`=%s WHERE gp_num=%s;"
                # 保存结果
                cursor = conn.cursor()
                print('写回数据库')
                print(result_json)
                cursor.execute(sql, args=[json.dumps(result_json), gp_num])
                conn.commit()
                json_res = {'code': 0}
                json_res.update(**make_res(train_loss_stat,
                                test_loss_stat, mape_stat, mae_stat, true_value, pred_value))
                return jsonify(json_res)
            except Exception as e:
                return jsonify({
                    'code': -1,
                    'message': str(e)
                })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
