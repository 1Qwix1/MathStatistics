import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Генерация данных
np.random.seed(42)
x = np.arange(-1.8, 2.2, 0.2)
n = len(x)
epsilon = np.random.normal(0, 1, n)
y = 2 + 2 * x + epsilon

# Данные с выбросами
y_outliers = y.copy()
y_outliers[0] += 10
y_outliers[-1] -= 10

# Метод наименьших квадратов
def ols(x, y):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.params

# Метод наименьших модулей
def lad(x, y):
    X = sm.add_constant(x)
    def lad_loss(params):
        return np.sum(np.abs(y - X @ params))
    res = minimize(lad_loss, [0, 0])
    return res.x

# Вычисление коэффициентов
ols_params = ols(x, y)
lad_params = lad(x, y)
ols_params_outliers = ols(x, y_outliers)
lad_params_outliers = lad(x, y_outliers)

# Построение графиков
plt.figure(figsize=(14, 7))

# График без выбросов
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Данные')
plt.plot(x, 2 + 2 * x, label='Истинная модель', color='blue', linestyle='dashed')
plt.plot(x, ols_params[0] + ols_params[1] * x, label='МНК', color='red')
plt.plot(x, lad_params[0] + lad_params[1] * x, label='МНМ', color='green')
plt.title('Без выбросов')
plt.legend()

# График с выбросами
plt.subplot(1, 2, 2)
plt.scatter(x, y_outliers, label='Данные с выбросами')
plt.plot(x, 2 + 2 * x, label='Истинная модель', color='blue', linestyle='dashed')
plt.plot(x, ols_params_outliers[0] + ols_params_outliers[1] * x, label='МНК', color='red')
plt.plot(x, lad_params_outliers[0] + lad_params_outliers[1] * x, label='МНМ', color='green')
plt.title('С выбросами')
plt.legend()

plt.show()

# Вывод коэффициентов
print("МНК коэффициенты без выбросов:", ols_params)
print("МНМ коэффициенты без выбросов:", lad_params)
print("МНК коэффициенты с выбросами:", ols_params_outliers)
print("МНМ коэффициенты с выбросами:", lad_params_outliers)