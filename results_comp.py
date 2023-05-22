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
    # multiplier = 0
    # width = 0.8
    # x = np.arange(9)
    # methods = []
    # offset = 0
    # colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']
    # for file in sorted(os.listdir('./download_1')):
    #     if '.csv' in file:
    #         df = pd.read_csv(os.path.join('./download_1', file), encoding = "ISO-8859-1")
    #         methods.append(file.split('- ')[1][:-4])
    #         offset += 1
    #         errors = df['Errors_types'].unique()
    #         rects = []
    #         for err, color in zip(errors, colors):
    #             offset += 1
    #             print(file, err)
    #             print(df[df['Errors_types'] == err]['F1_score'].max())
    #             rects.append(ax.bar(offset, df[df['Errors_types'] == err]['F1_score'].max(), width,
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
    # plt.ylim(0.8, 1.02)
    # plt.ylabel('F1 score')
    # plt.xticks(rotation=14)
    # plt.savefig('models_3c.svg')
    # plt.show()

    # different clustering
    # multiplier = 0
    # width = 0.6
    # x = np.arange(9)
    # # methods = []
    # offset = 0
    # colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']
    # for file in sorted(os.listdir('./download_1')):
    #     if 'Clustering_test.csv' in file:
    #         df = pd.read_csv(os.path.join('./download_1', file), encoding="ISO-8859-1")
    #         # methods.append(file.split('- ')[1][:-4])
    #         df['model_pret'] = (df['Model'] + ' ' + df['Description'])
    #         df['model_pret'] = df['model_pret'].str.replace(' no pretraining', '')
    #         errors = df['Errors_types'].unique()
    #         models = df['model_pret'].unique()
    #         for model in models:
    #             rects = []
    #             offset += 1
    #             for err, color in zip(errors, colors):
    #                 offset += 1
    #                 print(file, err)
    #                 print(df[df['model_pret']==model][df['Errors_types'] == err]['F1_score'].max())
    #                 rects.append(ax.bar(offset, df[df['model_pret']==model][df['Errors_types'] == err]['Detection_rate'].max(), width,
    #                                     label=err.replace("[", '').replace("]", '').replace("'", ''), color=color))
    #                 # ax.bar_label(rects, padding=5)
    #                 multiplier += 1
    #
    # # methods = [m.replace('_test', '') for m in methods]
    # # print(methods)
    # ax.set_xticks(x + width * 8.33 * (x + 1) - 1, models)
    # ax.legend(handles=rects, loc='lower left', title='Rodzaj błędu')
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray')
    # # plt.ylim(0.5, 1.02)
    # plt.ylabel('Detection rate')
    # plt.xticks(rotation=-14, ha='left')
    # plt.savefig('models_3c_clustering_detection_rate.svg')
    # plt.show()

    # different reduciotn
    # multiplier = 0
    # width = 0.6
    # x = np.arange(3)
    # # methods = []
    # offset = 0
    # colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']
    # for file in sorted(os.listdir('./download_1')):
    #     if 'Clustering_test.csv' in file:
    #         df = pd.read_csv(os.path.join('./download_1', file), encoding="ISO-8859-1")
    #         # methods.append(file.split('- ')[1][:-4])
    #         df = df[df['Model']=='DBscan']
    #         errors = df['Errors_types'].unique()
    #         reduction = df['Reduction method'].unique()
    #         for red in reduction:
    #             rects = []
    #             offset += 1
    #             for err, color in zip(errors, colors):
    #                 offset += 1
    #                 print(file, err)
    #                 print(df[df['Reduction method']==red][df['Errors_types'] == err]['F1_score'].max())
    #                 rects.append(ax.bar(offset, df[df['Reduction method']==red][df['Errors_types'] == err]['F1_score'].max(), width,
    #                                     label=err.replace("[", '').replace("]", '').replace("'", ''), color=color))
    #                 # ax.bar_label(rects, padding=5)
    #                 multiplier += 1
    #
    # # methods = [m.replace('_test', '') for m in methods]
    # # print(methods)
    # ax.set_xticks(x + width * 8.33 * (x + 1) - 1, reduction)
    # ax.legend(handles=rects, loc='lower right', title='Rodzaj błędu')
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray')
    # plt.ylim(0.7, 1.02)
    # plt.ylabel('F1 score')
    # # plt.xticks(rotation=14)
    # plt.savefig('dbscan_different_reduction.svg')
    # plt.show()

    # 1 błąd, różne klastrowania i metody redukcji
    # multiplier = 0
    # width = 0.6
    # x = np.arange(8)
    # # methods = []
    # offset = 0
    # colors = ['tab:olive', 'tab:cyan', 'tab:pink']
    #
    # for file in sorted(os.listdir('./download_1')):
    #     if 'Clustering_test.csv' in file:
    #         df = pd.read_csv(os.path.join('./download_1', file), encoding="ISO-8859-1")
    #         # methods.append(file.split('- ')[1][:-4])
    #         df = df[df['Errors_types'] == "['erratic']"]
    #         df['model_pret'] = (df['Model'] + ' ' + df['Description'])
    #         df['model_pret'] = df['model_pret'].str.replace(' no pretraining', '')
    #         reduction = df['Reduction method'].unique()
    #         reduction = ['PCA', 'UMAP', 'TSNE']
    #         models = df['model_pret'].unique()
    #         models = models[models != 'Spectral clustering']
    #         for model in models:
    #             rects = []
    #             offset += 1
    #             for red, color in zip(reduction, colors):
    #                 offset += 1
    #                 print(file, red)
    #                 # print(df[df['Reduction method']==red][df['Errors_types'] == err]['F1_score'].max())
    #                 rects.append(ax.bar(offset, df[df['Reduction method']==red][df['model_pret'] == model]['F1_score'].max(), width,
    #                                     label=red.replace("[", '').replace("]", '').replace("'", ''), color=color))
    #                 # ax.bar_label(rects, padding=5)
    #                 multiplier += 1
    #
    # # methods = [m.replace('_test', '') for m in methods]
    # # print(methods)
    # ax.set_xticks(x + width * 5 * (x + 1), models)
    # ax.legend(handles=rects, loc='lower right', title='Metoda redukcji wymiarów')
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray')
    # # plt.ylim(0.7, 1.02)
    # plt.ylabel('F1 score')
    # plt.xticks(rotation=-14, ha='left')
    # plt.savefig('erratic_different_models_and_reduction.svg')
    # plt.show()

    # błędy w różnych kolumnach
    # multiplier = 0
    # width = 0.8
    # x = np.arange(8)
    # methods = []
    # offset = 0
    # colors = ['lawngreen', 'dodgerblue', 'crimson']
    # for file in sorted(os.listdir('./download_1')):
    #     if '.csv' in file and 'Clustering' not in file:
    #         df = pd.read_csv(os.path.join('./download_1', file), encoding = "ISO-8859-1")
    #         methods.append(file.split('- ')[1][:-4])
    #         offset += 1
    #         # df = df[df['Errors_types']=="['erratic']"]
    #         # df = df[df['Errors_types'] == "['hardover']"]
    #         # df = df[df['Errors_types'] == "['spike']"]
    #         df = df[df['Errors_types'] == "['drift']"]
    #         cols = df['Error_detected'].unique()
    #         rects = []
    #         for col, color in zip(cols, colors):
    #             offset += 1
    #             print(file, col)
    #             print(df[df['Error_detected'] == col]['F1_score'].max())
    #             rects.append(ax.bar(offset, df[df['Error_detected'] == col]['F1_score'].max(), width,
    #                            label=col.replace("[", '').replace("]", '').replace("'", ''), color=color))
    #             # ax.bar_label(rects, padding=5)
    #             multiplier += 1
    #
    # methods = [m.replace('_test', '') for m in methods]
    # print(methods)
    # ax.set_xticks(x + width*3.75*(x+1), methods)
    # ax.legend(handles=rects, loc='lower right', title='Parametr dla którego następuje detekcja błedu')
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray')
    # plt.ylim(0.8, 1.02)
    # plt.ylabel('F1 score')
    # plt.xticks(rotation=14)
    # plt.savefig('drift_in_diff_columns.eps')
    # plt.show()

    # podział na zbiory
    # multiplier = 0
    # width = 0.8
    # x = np.arange(9)
    # methods = []
    # offset = 0
    # colors = ['orangered', 'gold', 'greenyellow', 'turquoise', 'mediumorchid']
    # for file in sorted(os.listdir('./download_1')):
    #     if '.csv' in file:
    #         df = pd.read_csv(os.path.join('./download_1', file), encoding = "ISO-8859-1")
    #         methods.append(file.split('- ')[1][:-4])
    #         offset += 1
    #         df = df[df['Errors_types']=="['erratic']"]
    #         datasets = df['Filename'].unique()
    #         rects = []
    #         for dataset, color in zip(datasets, colors):
    #             offset += 1
    #             print(file, dataset)
    #             print(df[df['Filename'] == dataset]['F1_score'].max())
    #             rects.append(ax.bar(offset, df[df['Filename'] == dataset]['Detection_rate'].max(), width,
    #                            label=dataset.replace("[", '').replace("]", '').replace("'", ''), color=color))
    #             # ax.bar_label(rects, padding=5)
    #             multiplier += 1
    # methods = [m.replace('_test', '') for m in methods]
    # print(methods)
    # ax.set_xticks(x + width*6.25*(x+1) - 1, methods)
    # ax.legend(handles=rects, loc='lower right')
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray')
    # # plt.ylim(0.7, 1.02)
    # plt.ylabel('Detection rate')
    # plt.xticks(rotation=14)
    # plt.savefig('different_datasets_erratic_dr.eps')
    # plt.show()

    # auc
