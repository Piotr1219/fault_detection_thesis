import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    # fig, ax = plt.subplots()
    # multiplier = 0
    width = 2
    # x = np.arange(9)
    methods = []
    scores = []
    offset = 0
    # colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']
    for file in sorted(os.listdir('./download_1')):
        if '.csv' in file:
            df = pd.read_csv(os.path.join('./download_1', file), encoding="ISO-8859-1")
            methods.append(file.split('- ')[1][:-4])
            errors = df['Errors_types'].unique()
            # rects = []
            # v = 0
            vd = 0
            for i, err in enumerate(errors):
                # v += df[df['Errors_types'] == err]['F1_score'].max()
                vd += df[df['Errors_types'] == err]['Detection_rate'].max()
                if i > 3:
                    # v /= 5
                    vd /= 5
                    scores.append(vd)
    #
    methods = [m.replace('_test', '') for m in methods]
    # print(methods)
    res = np.array([methods, scores])
    res = res[:, res[1, :].argsort()]
    # for i in range(len(res[0])):
    #     rects = []
    #     print(res[1][i])
    #     rects.append(ax.bar(offset, res[1][i], width,
    #                label='Detection rate', color='tab:green'))
    #     offset += 6
    # # ax.bar(res[0, :], res[1, :], width=0.2)
    # methods = res[0, :]
    # print(methods)
    #
    # # ax.set_xticks(x + width*3.15*(x+1) - 0.2, methods)
    # # ax.legend(handles=rects, loc='lower right')
    # # ax.set_axisbelow(True)
    # # ax.yaxis.grid(color='gray')
    # plt.ylim(0.2, 5.02)
    # # ax.set_ylim(0, 1.02)
    # # plt.ylabel('Detection rate')
    # # plt.xticks(rotation=14)
    # # plt.savefig('models_3c_mean_dr.eps')
    # plt.show()

    print(res[0])
    print(type(res[1][0]))
    fig, ax = plt.subplots()
    ax.bar(res[0], res[1].astype(float))
    plt.show()
