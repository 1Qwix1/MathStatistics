import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import factorial

def normal_pdf(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def student_pdf(x, df):
    return math.gamma((df + 1) / 2) / (np.sqrt(df * np.pi) * math.gamma(df / 2)) * (1 + (x ** 2) / df) ** (-(df + 1) / 2)

def poisson_pmf(x, base):
    return np.exp(-base) * np.power(base, x) / factorial(x)

def cauchy_pdf(x):
    return 1 / (np.pi * (x ** 2 + 1))

def uniform_pdf(x, a, b):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

def plot_distribution(title, data, distribution, params):
    for i in range(len(sample_sizes)):
        if i == 0:
            bins = 8;
        elif i == 1:
            bins = 16;
        elif i == 2:
            bins = 32;
        plt.subplot(1, 3, i + 1)  
        count, bins, ignored = plt.hist(data[i], bins, density=True, edgecolor='black')
        plt.plot(bins, distribution(bins, *params), linewidth=2, color='r')
        plt.title(f'{title} N={sample_sizes[i]}')
        plt.xlabel(f'{title} Numbers')
        plt.ylabel('Density')
        plt.subplots_adjust(wspace=0.5)
    plt.show()

sample_sizes = [10, 50, 1000]

    # Normal Distribution
mu, sigma = 0, 1
data_normal = [np.random.normal(mu, sigma, n) for n in sample_sizes]
plot_distribution('Normal', data_normal, normal_pdf, (mu, sigma))

# Cauchy Distribution
data_cauchy = [np.random.standard_cauchy(n) for n in sample_sizes]
plot_distribution('Cauchy', data_cauchy, cauchy_pdf, ())

# Student's t Distribution
df = 3  # degrees of freedom
data_student = [np.random.standard_t(df, n) for n in sample_sizes]
plot_distribution("Student's t", data_student, student_pdf, (df,))


# Poisson Distribution
base = 10
data_poisson = [np.random.poisson(base, n) for n in sample_sizes]
plot_distribution('Poisson', data_poisson, poisson_pmf, (base,))


# Uniform Distribution
a, b = -np.sqrt(3), np.sqrt(3)
data_uniform = [np.random.uniform(a, b, n) for n in sample_sizes]
plot_distribution('Uniform', data_uniform, uniform_pdf, (a, b))