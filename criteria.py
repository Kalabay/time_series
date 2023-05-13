import numpy as np

def ordinal_patterns(window_size: int, time_series: np.ndarray):
    '''
    Реализация метода ordinal patterns для перевода временного ряда в массив вероятностей

    Parameters
    ----------
    window_size : int
        Размер скользящего окна
    time_series : np.ndarray
        Временной ряд

    Возвращает массив вероятностей
    '''
    patterns = []
    for start in range(time_series.shape[0] - window_size + 1):
        patterns.append((np.argsort(time_series[start:start + window_size])).tolist())
    patterns.sort()
    counts = []
    count_now = 0
    last_pattern = []
    for pattern in patterns:
        if pattern != last_pattern:
            last_pattern = pattern
            counts.append(count_now)
            count_now = 1
        else:
            count_now += 1
    counts = np.array(counts + [count_now])
    probabilities = counts / len(patterns)
    return probabilities


def shannon_entropy(probabilities: np.ndarray):
    '''
    Подсчёт энтропии Шеннона

    Parameters
    ----------
    probabilities : np.ndarray
        Массив вероятностей

    Возвращает энтропию Шеннона
    '''
    probabilities = probabilities[probabilities != 0]
    return -(np.log(probabilities) * probabilities).sum()


def q_0(count_all: int):
    '''
    Подбор параметра для нормализации сложности

    Parameters
    ----------
    count_all : int
        Количество возможных паттернов

    Возвращает коэффициент нормализации
    '''
    return 1 / (np.log(2 * count_all) + (count_all + 1)/(2 * count_all) * -np.log(count_all + 1) - np.log(count_all) / 2)


def complexity(count_all: int, sh_entropy: float, probabilities: np.ndarray):
    '''
    Подбор параметра для нормализации сложности

    Parameters
    ----------
    count_all : int
        Количество возможных паттернов
    sh_entropy : float
        Энтропия Шеннона
    probabilities : np.ndarray
        Массив вероятностей

    Возвращает сложность
    '''
    q_j = 0
    q_j += shannon_entropy((probabilities + 1 / count_all) / 2)
    q_j += (count_all - probabilities.shape[0]) / 2 / count_all * np.log(2 * count_all)
    q_j -= sh_entropy / 2
    q_j -= np.log(count_all) / 2
    return q_j * q_0(count_all)


def entropy_complexity(n: int, time_series: np.ndarray):
    '''
    Метод для подсчёта энтропии и сложности

    Parameters
    ----------
    n : int
        Размер паттернов
    time_series : np.ndarray
        Временной ряд

    Возвращает энтропию и сложность
    '''
    probabilities = ordinal_patterns(n, time_series)
    count_all = np.math.factorial(n)
    sh_entropy = shannon_entropy(probabilities)
    q_j = complexity(count_all, sh_entropy, probabilities)
    sh_entropy /= np.log(count_all)
    c_js = q_j * sh_entropy
    return sh_entropy, c_js