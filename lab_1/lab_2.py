import numpy as np
from scipy import stats

# Функции для вычисления статистических характеристик
def mean(arr):
    return sum(arr) / len(arr)

def median(arr):
    if len(arr) % 2 == 1: 
        return arr[len(arr) // 2]
    else:
        return 0.5 * (arr[len(arr) // 2] + arr[len(arr) // 2 + 1])

def zr(arr):
    return 0.5*(np.max(arr) + np.min(arr))

def zq(arr):
    q1 = np.percentile(arr, 25) #Вычислите q-й процентиль данных вдоль указанной оси.
    q3 = np.percentile(arr, 75) # Вычислите q-й процентиль данных вдоль указанной оси.
    return 0.5 * (q1 + q3)

def ztr(arr):
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    k = int(0.25 * n)
    return np.mean(arr_sorted[k:n-k])

# Генерация выборок и вычисление статистических характеристик
distributions = {
    "Нормальное": np.random.normal,
    "Коши": np.random.standard_cauchy,
    "Стюдента": lambda size: np.random.standard_t(df=3, size=size),
    "Пуассона": lambda size: np.random.poisson(lam=10, size=size),
    "Равномерное": lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size=size)
}

sample_sizes = [10, 100, 1000]
results = {}

for name, distribution in distributions.items():
    results[name] = {}
    for size in sample_sizes:
        results[name][size] = {}
        for _ in range(1000):
            sample = distribution(size=size)
            results[name][size][_+1] = {
                "mean": mean(sample),
                "median": median(sample),
                "zr": zr(sample),
                "zq": zq(sample),
                "ztr": ztr(sample)
            }

# Вычисление средних и квадратов характеристик положения
means = {}
squares = {}

for name in distributions.keys():
    means[name] = {}
    squares[name] = {}
    for size in sample_sizes:
        means[name][size] = {}
        squares[name][size] = {}
        for stat in ["mean", "median", "zr", "zq", "ztr"]:
            means[name][size][stat] = np.mean([results[name][size][i][stat] for i in range(1, 1001)])
            squares[name][size][stat] = np.mean([((results[name][size][i][stat]**2) - means[name][size][stat]**2) for i in range(1, 1001)])

# Вывод результатов в виде таблицы
print("Средние характеристики положения:")
for name in distributions.keys():
    print(f"\n{name}:")
    for size in sample_sizes:
        print(f"Размер выборки {size}:")
        for stat in ["mean", "median", "zr", "zq", "ztr"]:
            print(f"{stat}: {means[name][size][stat]}")

print("\nКвадраты характеристик положения:")
for name in distributions.keys():
    print(f"\n{name}:")
    for size in sample_sizes:
        print(f"Размер выборки {size}:")
        for stat in ["mean", "median", "zr", "zq", "ztr"]:
            print(f"{stat}: {squares[name][size][stat]}")