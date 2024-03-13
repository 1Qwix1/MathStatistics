import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.special as sc

def draw_boxplot(data, labels, title):
    line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
    bbox_props = dict(color="r", alpha=0.9)
    flier_props = dict(marker="o", markersize=4)
    plt.boxplot(data, whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=labels)
    plt.ylabel("X")
    plt.title(title)
    plt.show()

X1 = np.random.normal(0, 1, 20)
X2 = np.random.normal(0, 1, 100)
draw_boxplot((X1, X2), ["n = 20", "n = 100"], "Normal")

X1 = np.random.standard_cauchy(20)
X2 = np.random.standard_cauchy(100)
draw_boxplot((X1, X2), ["n = 20", "n = 100"], "Cauchy")

X1 = np.random.standard_t(3, 20)
X2 = np.random.standard_t(3, 100)
draw_boxplot((X1, X2), ["n = 20", "n = 100"], "Student")

X1 = np.random.poisson(10, 20)
X2 = np.random.poisson(10, 100)
draw_boxplot((X1, X2), ["n = 20", "n = 100"], "Poisson")

X1 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 20)
X2 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 100)
draw_boxplot((X1, X2), ["n = 20", "n = 100"], "Uniform")