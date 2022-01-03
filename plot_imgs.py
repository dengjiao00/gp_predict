import numpy as np
from matplotlib import pyplot as plt


def plot_more(result_list):
    net_name_list = []
    num_epoches_list = []
    train_loss_list_list = []
    test_loss_list_list = []
    mae_list_list = []
    rmse_list_list = []
    plot_predict_list = []
    for result in result_list:
        net_name = result['net_name']
        num_epoches = result['num_epoches']
        train_loss_list = result['train_loss_list']
        test_loss_list = result['test_loss_list']
        mae_list = result['mae_list']
        rmse_list = result['rmse_list']
        plot_predict = result['plot_predict']

        net_name_list.append(net_name)
        num_epoches_list.append(num_epoches)
        train_loss_list_list.append(train_loss_list)
        test_loss_list_list.append(test_loss_list)
        mae_list_list.append(mae_list)
        rmse_list_list.append(rmse_list)
        plot_predict_list.append(plot_predict)


    # 画出MAE图像
    for idx, mae_list in enumerate(mae_list_list):
        plt.plot(range(1, num_epoches + 1), mae_list, label=f'mae {idx}')
    plt.title('MAE')
    plt.xlabel('mae')
    plt.ylabel('epoch')
    plt.legend()
    plt.show()

    # 画出RMSE图像
    for idx, rmse_list in enumerate(rmse_list_list):
        plt.plot(range(1, num_epoches + 1), rmse_list, label=f'rmse {idx}')
    plt.title('RMSE')
    plt.xlabel('rmse')
    plt.ylabel('epoch')
    plt.legend()
    plt.show()


    for idx, plot_predict in enumerate(plot_predict_list):
        x_len = len(plot_predict)
        plt.plot(range(1, x_len + 1), plot_predict, label=f'true value {idx}')
        plt.title('true/pred')
        plt.legend()
        plt.show()