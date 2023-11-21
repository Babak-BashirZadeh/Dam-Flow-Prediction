import matplotlib.pyplot as plt
import numpy as np

def plot_curve(data , labels , color , x_label , y_label):
    plt.figure(figsize=(10,5))

    for i in range(len(data)) :
        curve = data[i]
        lbl = labels[i]
        clr = color[i]
        plt.plot(np.arange(curve.shape[0]),curve , clr , label=lbl)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    plt.legend()
    plt.show()