import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(12, 8))
    # for file in os.listdir('./download_1'):
    #     if '.csv' in file:
    #         df = pd.read_csv(os.path.join('./download_1', file), encoding = "ISO-8859-1")
    #         plt.plot(df.groupby('Errors_types')['F1_score'].max().index, df.groupby('Errors_types')['F1_score'].max(),
    #                  label=file.split('- ')[1][:-4])

    # different models
    multiplier = 0
    width = 0.8
    x = np.arange(9)
    methods = []
    offset = 0
    colors = ['deepskyblue', 'purple']
    diff = []
    same = []
    err = "['erratic']"
    for file in sorted(os.listdir('./download_2c_different')):
        if '.csv' in file:
            df = pd.read_csv(os.path.join('./download_2c_different', file), encoding = "ISO-8859-1")
            methods.append(file.split('- ')[1][:-4])
            df = df[df['Errors_types'] == err]
            diff.append(df['F1_score'].max())
    for file in sorted(os.listdir('./download_2c_same')):
        if '.csv' in file:
            df = pd.read_csv(os.path.join('./download_2c_same', file), encoding="ISO-8859-1")
            df = df[df['Errors_types'] == err]
            same.append(df['F1_score'].max())

    for i in range(len(diff)):
        offset += 2
        rects = []
        rects.append(ax.bar(offset, diff[i], width,
                           label="Różne okresy", color=colors[0]))
        offset += 1
        rects.append(ax.bar(offset, same[i], width,
                            label="Te same okresy", color=colors[1]))
        multiplier += 1

    methods = [m.replace('_test', '') for m in methods]
    print(methods)
    ax.set_xticks(x + width*2.5*(x+1) + 0.5, methods)
    ax.legend(handles=rects, loc='lower right', title='Występowanie błędów')
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray')
    plt.ylim(0.8, 1.02)
    plt.ylabel('F1 score')
    plt.xticks(rotation=14)
    plt.savefig('models_2c_same_diff.eps')
    plt.show()

    # multiplier = 0
    # width = 0.8
    # x = np.arange(9)
    # methods = []
    # offset = 0
    # colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']
    # for file in sorted(os.listdir('./download_2c_different')):
    #     if '.csv' in file:
    #         df = pd.read_csv(os.path.join('./download_2c_different', file), encoding = "ISO-8859-1")
    #         methods.append(file.split('- ')[1][:-4])
    #         offset += 1
    #         errors = df['Errors_types'].unique()
    #         rects = []
    #         for err, color in zip(errors, colors):
    #             offset += 1
    #             print(file, err)
    #             print(df[df['Errors_types'] == err]['F1_score'].max())
    #             rects.append(ax.bar(offset, df[df['Errors_types'] == err]['Detection_rate'].max(), width,
    #                            label=err.replace("[", '').replace("]", '').replace("'", ''), color=color))
    #             # ax.bar_label(rects, padding=5)
    #             multiplier += 1
    #
    # methods = [m.replace('_test', '') for m in methods]
    # print(methods)
    # ax.set_xticks(x + width*6.25*(x+1) - 1, methods)
    # ax.legend(handles=rects, loc='lower right')
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray')
    # # plt.ylim(0.8, 1.02)
    # plt.ylabel('Detection rate')
    # plt.xticks(rotation=14)
    # plt.savefig('models_2c_diff_dr.eps')
    # plt.show()
