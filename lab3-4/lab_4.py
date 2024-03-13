import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Доверительный интервал для выборочного среднего (нормальное распределение)
def ci_mean_norm(sample, alpha=0.05):

    sample_mean = np.mean(sample)
    sample_std = np.std(sample)
    sample_size = len(sample)

    z_score = stats.norm.ppf(1 - alpha / 2)
    margin_of_error = z_score * (sample_std / np.sqrt(sample_size))

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return lower_bound, upper_bound

# Доверительный интервал для выборочного среднего (произвольное распределение)
def ci_mean_t(sample, alpha=0.05):

    sample_mean = np.mean(sample)
    sample_std = np.std(sample)
    sample_size = len(sample)

    t_score = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)
    margin_of_error = t_score * (sample_std / np.sqrt(sample_size))

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return lower_bound, upper_bound

# Доверительный интервал для выборочной дисперсии (нормальное распределение)
def ci_var_norm(sample, alpha=0.05):

    sample_var = np.var(sample)
    sample_size = len(sample)

    chi2_lower = stats.chi2.ppf(alpha / 2, df=sample_size - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=sample_size - 1)

    lower_bound = (sample_size - 1) * sample_var / chi2_upper
    upper_bound = (sample_size - 1) * sample_var / chi2_lower

    return lower_bound, upper_bound

# Доверительный интервал для выборочной дисперсии (произвольное распределение)
def ci_var_t(sample, alpha=0.05):

    sample_var = np.var(sample)
    sample_size = len(sample)

    chi2_lower = stats.chi2.ppf(alpha / 2, df=sample_size - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=sample_size - 1)

    lower_bound = (sample_size - 1) * sample_var / chi2_upper
    upper_bound = (sample_size - 1) * sample_var / chi2_lower

    return lower_bound, upper_bound

# Генерация данных
sample1 = np.random.normal(size=100)

sample2 = np.random.poisson(10, size=100)
# Доверительные интервалы
ci_mean_normal = ci_mean_norm(sample1, 0.95)
ci_mean_t = ci_mean_t(sample2, 0.95)
ci_var_normal = ci_var_norm(sample1, 0.95)
ci_var_t = ci_var_t(sample2, 0.95)

# Гистограммы
plt.hist(sample1, density=True)
plt.axvline(ci_mean_normal[0], color='red', linestyle='--')
plt.axvline(ci_mean_normal[1], color='red', linestyle='--')
plt.axvline(ci_mean_normal[0] - ci_var_normal[0], color='blue', linestyle='--')
plt.axvline(ci_mean_normal[1] + ci_var_normal[1], color='blue', linestyle='--')
plt.title('Гистограмма нормального распределения с доверительными интервалами n = 100')
plt.show()

plt.hist(sample2, density=True)
plt.axvline(ci_mean_t[0], color='red', linestyle='--')
plt.axvline(ci_mean_t[1], color='red', linestyle='--')
plt.axvline(ci_mean_t[0] - ci_var_t[0], color='blue', linestyle='--')
plt.axvline(ci_mean_t[1] + ci_var_t[1], color='blue', linestyle='--')
plt.title('Гистограмма произвольного распределения с доверительными интервалами n = 100')
plt.show()



# Генерация данных
sample1 = np.random.normal(size=20)
sample2 = np.random.normal(size=100)

# Доверительные интервалы
ci_std_normal_20 = ci_var_norm(sample1, 0.95)
ci_std_normal_100 = ci_var_norm(sample2, 0.95)

# График
plt.plot([20, 100], [ci_std_normal_20[0], ci_std_normal_100[0]], color='blue', marker='o', label='Нижняя граница')
plt.plot([20, 100], [ci_std_normal_20[1], ci_std_normal_100[1]], color='red', marker='o', label='Верхняя граница')
plt.xlabel('Размер выборки')
plt.ylabel('Доверительный интервал для СКО')
plt.title('Доверительные интервалы для СКО в зависимости от размера выборки (нормальное распределение)')
plt.legend()
plt.show()

# Генерация данных
sample1 = np.random.poisson(10 ,size=20)
sample2 = np.random.poisson(10 , size=100)

# Доверительные интервалы
ci_std_normal_20 = ci_var_norm(sample1, 0.95)
ci_std_normal_100 = ci_var_norm(sample2, 0.95)

# График
plt.plot([20, 100], [ci_std_normal_20[0], ci_std_normal_100[0]], color='blue', marker='o', label='Нижняя граница')
plt.plot([20, 100], [ci_std_normal_20[1], ci_std_normal_100[1]], color='red', marker='o', label='Верхняя граница')
plt.xlabel('Размер выборки')
plt.ylabel('Доверительный интервал для СКО')
plt.title('Доверительные интервалы для СКО в зависимости от размера выборки (произвольное распределение)')
plt.legend()
plt.show()

print("Доверительный интервал для среднего (нормальное распределение):", ci_mean_normal)
print("Доверительный интервал для среднего (произвольное распределение):", ci_mean_t)
print("Доверительный интервал для дисперсии (нормальное распределение):", ci_var_normal)
print("Доверительный интервал для дисперсии (произвольное распределение):", ci_var_t)