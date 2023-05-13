import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import KDTree


class Wishart:
    def __init__(self, n_neighbors: int, marginal_value: float):
        '''
        Класс - алгоритм Уишарта

        Parameters
        ----------
        n_neighbors : int
            Количество соседей
        marginal_value : float
            Параметр для сравнения
        '''
        self.n_neighbors = n_neighbors
        self.marginal_value = marginal_value

    def norm(self, cluster_number: np.ndarray):
        '''
        Перенумерация кластеров

        Parameters
        ----------
        cluster_number : np.ndarray
            Информация о номере кластера,
            которому принадлежит объект

        Возвращает новые номера
        '''
        clusters_name = np.unique(cluster_number)
        new_name = dict()
        check_zero = 1 - np.any(np.isin(0, clusters_name))
        for i in range(clusters_name.shape[0]):
            new_name[clusters_name[i]] = i + check_zero
        new_clusters = []
        for i in range(cluster_number.shape[0]):
            new_clusters.append(new_name[cluster_number[i]])
        return np.array(new_clusters)

    def fit_predict(self, X):
        '''
        Обучение, получаем номер кластера
        для каждого объекта

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка

        Возвращает номера кластеров
        '''
        n, dimension = X.shape
        cluster_number = -1 * np.ones(n, dtype=np.int32)
        clusters = [[]]
        n_cluster = 0
        kdt = KDTree(X, metric='euclidean')
        distances, neighbors = kdt.query(X, k=self.n_neighbors + 1, return_distance=True)
        neighbors = neighbors[:, 1:]
        distances = distances[:, -1]
        self.dk = distances
        self.dist = np.array(squareform(pdist(X)))
        t = dimension / 2
        volumes = np.pi ** t * distances ** dimension / gamma(t + 1)
        significance = distances / volumes / dimension

        order = np.argsort(distances)
        is_used = [False] * n
        is_completed = [False]
        is_significant = [False]
        minimum = [-1]
        maximum = [-1]
        for q in range(n):
            reachable_clusters = set()
            i = order[q]
            for j in neighbors[i]:
                if is_used[j]:
                    reachable_clusters.add(cluster_number[j])
            is_used[i] = True
            reachable_clusters = sorted(list(reachable_clusters))
            if len(reachable_clusters) == 0:
                cluster_number[i] = n_cluster + 1
                n_cluster += 1
                clusters.append([i])
                is_completed.append(False)
                is_significant.append(False)
                minimum.append(significance[i])
                maximum.append(significance[i])
            elif len(reachable_clusters) == 1:
                index_cluster = reachable_clusters[0]
                if is_completed[index_cluster]:
                    cluster_number[i] = 0
                    clusters[0].append(i)
                else:
                    cluster_number[i] = index_cluster
                    clusters[index_cluster].append(i)
                    minimum[index_cluster] = min(minimum[index_cluster], significance[i])
                    maximum[index_cluster] = max(maximum[index_cluster], significance[i])
                    if maximum[index_cluster] - minimum[index_cluster] >= self.marginal_value:
                        is_significant[index_cluster] = True
            else:
                check = True
                k = 0
                for cluster in reachable_clusters:
                    check = min(check, is_completed[cluster])
                    k += is_significant[cluster]
                if check:
                    cluster_number[i] = 0
                    clusters[0].append(i)
                elif k > 1 or reachable_clusters[0] == 0:
                    cluster_number[i] = 0
                    clusters[0].append(i)
                    for cluster in reachable_clusters:
                        if is_significant[cluster]:
                            is_completed[cluster] = True
                        elif cluster != 0:
                            for j in clusters[cluster]:
                                clusters[0].append(j)
                                cluster_number[j] = 0
                            clusters[cluster] = []
                else:
                    index_cluster = reachable_clusters[0]
                    for cluster in reachable_clusters[1:]:
                        for j in clusters[cluster]:
                            clusters[index_cluster].append(j)
                            cluster_number[j] = index_cluster
                            minimum[index_cluster] = min(minimum[index_cluster], significance[j])
                            maximum[index_cluster] = max(maximum[index_cluster], significance[j])
                            if maximum[index_cluster] - minimum[index_cluster] >= self.marginal_value:
                                is_significant[index_cluster] = True
                        clusters[cluster] = []
                    cluster_number[i] = index_cluster
                    clusters[index_cluster].append(i)
                    minimum[index_cluster] = min(minimum[index_cluster], significance[i])
                    maximum[index_cluster] = max(maximum[index_cluster], significance[i])
                    if maximum[index_cluster] - minimum[index_cluster] >= self.marginal_value:
                        is_significant[index_cluster] = True
        self.cluster_number = self.norm(cluster_number)
        return self.cluster_number

    def count_clasters(self):
        '''
        Количество кластеров
        '''
        return np.max(self.cluster_number)

    def is_noise(self):
        '''
        Проверка на наличие шума
        '''
        return np.min(self.cluster_number) == 0