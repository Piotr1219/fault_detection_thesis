import matplotlib.pyplot as plt

def bar_plot():
    fig, ax = plt.subplots()

    methods = ['DBscan', 'Isolation forests', 'Local outlier factor', 'Gaussian mixture']
    F1_no_pretraining = [80, 86, 95.5, 55.5]
    F1_pretraining = [80, 82.7, 95.4, 55.8]

    rects1 = ax.bar(methods, F1_no_pretraining, width=0.35)

    ax.set_ylabel('F1 score [%]')
    ax.set_title('F1 score without pretraining')
    # ax.set_title('F1 score with pretraining')
    ax.set_ylim(0, 105)
    ax.bar_label(rects1, padding=3)

    plt.show()

if __name__ == '__main__':
    bar_plot()

