import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as stats
from matplotlib.patches import Ellipse
from tabulate import tabulate
import statistics

# Load the data
data = pd.read_csv('0.05V_sp953.dat', sep=" ", header=None)
data.columns = [f'Var{i}' for i in range(data.shape[1])]

def comp_quadrant(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    x_new = x - med_x
    y_new = y - med_y
    n = [0, 0, 0, 0]
    for i in range(size):
        if x_new[i] >= 0 and y_new[i] >= 0:
            n[0] += 1
        if x_new[i] < 0 and y_new[i] > 0:
            n[1] += 1
        if x_new[i] < 0 and y_new[i] < 0:
            n[2] += 1
        if x_new[i] > 0 and y_new[i] < 0:
            n[3] += 1
    return ((n[0] + n[2]) - (n[1] + n[3])) / size

def correlation_coef(data, repeat):
    size = data.shape[0]
    quadrant_coef = np.empty(repeat, dtype=float)
    pearson_coef = np.empty(repeat, dtype=float)
    spearman_coef = np.empty(repeat, dtype=float)
    for i in range(repeat):
        sample = data.sample(n=size, replace=True).values
        x = sample[:, 0]
        y = sample[:, 1]
        quadrant_coef[i] = comp_quadrant(x, y)
        pearson_coef[i], _ = stats.pearsonr(x, y)
        spearman_coef[i], _ = stats.spearmanr(x, y)
    return quadrant_coef, pearson_coef, spearman_coef

def create_table(quadrant_coef, pearson_coef, spearman_coef):
    rows = []
    headers = ["Metric", "Pearson", "Spearman", "Quadrant"]
    q = np.mean(quadrant_coef)
    p = np.mean(pearson_coef)
    s = np.mean(spearman_coef)
    rows.append(['Mean', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    quadrant_coef_quad = quadrant_coef**2
    pearson_coef_quad = pearson_coef**2
    spearman_coef_quad = spearman_coef**2
    q = np.mean(quadrant_coef_quad)
    p = np.mean(pearson_coef_quad)
    s = np.mean(spearman_coef_quad)
    rows.append(['Mean of Squares', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    q = statistics.variance(quadrant_coef)
    p = statistics.variance(pearson_coef)
    s = statistics.variance(spearman_coef)
    rows.append(['Variance', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    print(tabulate(rows, headers, tablefmt="latex"))
    print('\n')

def set_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_ellipse(data, fig_num):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(f"Рис. {fig_num} Эллипс рассеивания")
    x = data['Var0']
    y = data['Var1']
    ax.scatter(x, y, s=3, color='darkorange')
    set_ellipse(x, y, ax, edgecolor='navy')
    ax.scatter(np.mean(x), np.mean(y), c='red', s=20)  # Red, smaller central point
    plt.show()

repeat = 1000
fig_num = 1
quadrant_coef, pearson_coef, spearman_coef = correlation_coef(data, repeat)
create_table(quadrant_coef, pearson_coef, spearman_coef)
plot_ellipse(data, fig_num)