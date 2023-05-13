import criteria
import lorentz
import wishart
import random
import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN

class TSProcessor:
    def __init__(self, points_in_template: int, max_template_spread: int, type_algo: str) -> None:
        '''
        Основной алгоритм предсказания

        Parameters
        ----------
        points_in_template : int
            Количество точек в шаблоне
        max_template_spread : int
            Макс. расстояне
        type_algo : str
            Тип алгоритма
        '''
        # максимальное расстояние между соседними зубчиками шаблона
        self._max_template_spread: int = max_template_spread
        self.x_dim: int = max_template_spread ** (points_in_template - 1)  # сколько у нас всего шаблонов
        self.z_dim: int = points_in_template                               # сколько зубчиков в каждом шаблоне
        self.type_algo: str = type_algo

        # сами шаблоны
        templates = (np.repeat(0, self.x_dim).reshape(-1, 1), )

        # заполняет шаблоны нужными значениями
        for i in range(1, points_in_template):
            col = (np.repeat(
                np.arange(1, max_template_spread + 1, dtype=int), max_template_spread ** (points_in_template - (i + 1))
            ) + templates[i - 1][::max_template_spread ** (points_in_template - i)]).reshape(-1, 1)

            templates += (col, )  # don't touch

        self._templates: np.ndarray = np.hstack(templates)

        # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        self._template_shapes: np.ndarray = self._templates[:, 1:] - self._templates[:, :-1]

        self.fits = []

    def fit_add(self, time_series: np.ndarray) -> None:
        '''Добавить ряд в обучающую выборку.'''
        _time_series   = time_series
        y_dim          = _time_series.size - self._templates[0][-1]

        # создать обучающее множество
        # Его можно представить как куб, где по оси X идут шаблоны, по оси Y - вектора,
        # а по оси Z - индивидуальные точки векторов.
        # Чтобы получить точку A вектора B шаблона C - делаем self._training_vectors[C, B, A].
        # Вектора идут в хронологическом порядке "протаскивания" конкретного шаблона по ряду,
        # шаблоны - по порядку от [1, 1, ... , 1], [1, 1, ..., 2] до [n, n, ..., n].
        _training_vectors: np.ndarray = \
            np.full(shape=(self.x_dim, y_dim, self.z_dim), fill_value=np.inf, dtype=float)

        # тащим шаблон по ряду
        for i in range(self.x_dim):
            template_data = (
                _time_series[self._templates[i]
                                  + np.arange(_time_series.size - self._templates[i][-1])[:, None]]
            )

            _training_vectors[i, :template_data.shape[0]] = (
                _time_series[self._templates[i]
                                  + np.arange(_time_series.size - self._templates[i][-1])[:, None]]
            )
        self.fits.append(_training_vectors)

    def fit(self, time_series: np.ndarray) -> None:
        '''Обучить класс на конкретном ряду.'''

        self._time_series   = time_series
        self.y_dim          = self._time_series.size - self._templates[0][-1]
        self._original_size = self._time_series.size

        # создать обучающее множество
        # Его можно представить как куб, где по оси X идут шаблоны, по оси Y - вектора,
        # а по оси Z - индивидуальные точки векторов.
        # Чтобы получить точку A вектора B шаблона C - делаем self._training_vectors[C, B, A].
        # Вектора идут в хронологическом порядке "протаскивания" конкретного шаблона по ряду,
        # шаблоны - по порядку от [1, 1, ... , 1], [1, 1, ..., 2] до [n, n, ..., n].
        self._training_vectors: np.ndarray = \
            np.full(shape=(self.x_dim, self.y_dim, self.z_dim), fill_value=np.inf, dtype=float)

        # тащим шаблон по ряду
        for i in range(self.x_dim):
            template_data = (
                self._time_series[self._templates[i]
                                  + np.arange(self._time_series.size - self._templates[i][-1])[:, None]]
            )

            self._training_vectors[i, :template_data.shape[0]] = (
                self._time_series[self._templates[i]
                                  + np.arange(self._time_series.size - self._templates[i][-1])[:, None]]
            )

    def pull(self, steps: int, eps: float, n_trajectories: int, noise_amp: float) -> np.ndarray:
        '''
        Основной метод пулла, который использовался в статье.

        Parameters
        ----------
        steps : int
            На сколько шагов прогнозируем.
        eps : float
            Минимальное Евклидово расстояние от соответствующего шаблона, в пределах которого должны находиться
            вектора наблюдений, чтобы считаться "достаточно похожими".
        n_trajectories : int
            Сколько траекторий использовать. Чем больше, тем дольше время работы и потенциально точнее результат.
        noise_amp : float
            Максимальная амплитуда шума, используемая при расчете траекторий.

        Возвращает матрицу размером steps x n_trajectories, где по горизонтали идут шаги, а по вертикали - прогнозы
        каждой из траекторий на этом шаге.
        '''

        # прибавляем к тренировочному датасету steps пустых векторов, которые будем заполнять значениями на ходу
        self._training_vectors = np.hstack([self._training_vectors,
                                            np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)])

        # удлиняем изначальый ряд на значение steps
        self._time_series          = np.resize(self._time_series, self._original_size + steps)
        self._time_series[-steps:] = np.nan

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        forecast_sets = np.full((steps, n_trajectories), np.nan)

        for i in range(n_trajectories):
            for j in range(steps):

                # тестовые вектора, которые будем сравнивать с тренировочными
                last_vectors = (self._time_series[:self._original_size + j]
                                                 [np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]])

                distance_matrix = self._calc_distance_matrix(last_vectors, np.repeat(True, self.x_dim), steps)

                # последние точки тренировочных векторов, оказавшихся в пределах eps
                points = self._training_vectors[distance_matrix < eps][:, -1]

                # теперь нужно выбрать финальное прогнозное значение из возможных
                # я выбираю самое часто встречающееся значение, но тут уже можно на свое усмотрение
                forecast_point                             = self._freeze_point(points, self.type_algo) \
                    + random.uniform(-noise_amp, noise_amp)
                forecast_sets[j, i]                        = forecast_point
                self._time_series[self._original_size + j] = forecast_point

                # у нас появилась новая точка в ряду, последние вектора обновились, добавим их в обучающие
                new_training_vectors = (
                    self._time_series[:self._original_size + 1 + j]
                    [np.hstack((np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]
                     - 1, np.repeat(-1, self.x_dim).reshape(-1, 1)))]
                )

                self._training_vectors[:, self.y_dim + j, :] = new_training_vectors

            # честно говоря, я не помню, зачем нужен код дальше

            # delete added vectors after each run
            self._training_vectors[:, self.y_dim:] = np.inf

            # delete added points after each run
            self._time_series[-steps:] = np.nan

        return forecast_sets

    def push_all(self, steps, eps, w_eps, w_neighbors, real_points, phi=1e9):
        '''Предсказать ряд на steps вперёд, используя несколько рядов'''
        self._training_vectors     = np.hstack([self._training_vectors,
                                                np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)])
        self._time_series          = np.resize(self._time_series, self._original_size + steps)
        self._time_series[-steps:] = np.nan
        point_pools = [[] for _ in range(steps)]

        for step in range(steps):
            for i in range(self._templates.shape[0]):
                pattern = (self._templates[i] * -1 + self._original_size + step)[::-1]
                if pattern[0] < 0:
                    continue
                block = self._time_series[pattern]
                if np.any(np.isnan(block[:-1])):
                    continue
                for j in range(len(self.fits)):
                    blocks = self.fits[j][i]
                    distance = np.sqrt(((blocks[:, :-1] - block[:-1]) ** 2).sum(axis=1))
                    for block in blocks[distance < eps]:
                        point_pools[step].append(block)
            point_pools[step] = np.array(point_pools[step])
            predict = self._freeze_point(point_pools[step], self.type_algo, w_eps, w_neighbors)
            #print(step, predict)
            if predict != np.nan and abs(predict - real_points[step]) <= phi:
                self._time_series[self._original_size + step] = predict
                for i in range(self._templates.shape[0]):
                    pattern = (self._templates[i] * -1 + self._original_size + step)[::-1]
                    if pattern[0] < 0:
                        continue
                    block = self._time_series[pattern]
                    if np.any(np.isnan(block)):
                        continue
                    self._training_vectors[i][pattern[0]] = block

        return self._time_series[-steps:]

    def cluster_sets(self, forecast_sets: np.ndarray, dbs_eps: float, dbs_min_samples: int):
        '''
        Скластеризировать полученные в результате пулла множества прогнозных значений.
        Возвращает центр самого большого кластера на каждом шаге.
        '''

        predictions = np.full(shape=[forecast_sets.shape[0], ], fill_value=np.nan)
        dbs         = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)

        for i in range(len(forecast_sets)):
            curr_set = forecast_sets[i]
            dbs.fit(curr_set.reshape(-1, 1))

            cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

            if cluster_labels.size > 0:
                biggest_cluster_center = curr_set[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                predictions[i] = biggest_cluster_center

        return predictions

    def _calc_distance_matrix(self, test_vectors: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray:
        '''
        Считает матрицу расстояний между тестовыми и тренировочными векторами.
        '''
        distance_matrix = np.zeros(self._training_vectors.shape[:2])

        for i in range(test_vectors.shape[1]):
            distance_matrix += (self._training_vectors[mask, :, i] - np.repeat(test_vectors[:, i], self.y_dim + steps)
                                .reshape(-1, self.y_dim + steps)) ** 2
        distance_matrix **= 0.5

        return distance_matrix

    def _freeze_point(self, points_pool: np.ndarray, how: str, w_eps: float = 0.0, w_neighbors: int = 0) -> float:
        '''
        Выбрать финальный прогноз в данной точке из множества прогнозных значений.

        Варианты:
            "mean" = "mean"
            "mf"   = "most frequent"
            "cl"   = "cluster", нужны dbs_eps и dbs_min_samples
        '''
        if points_pool.size == 0:
            result = np.nan
        else:
            if how == 'mean':
                result = float(points_pool[:, -1].mean())
            elif how == 'mf':
                points, counts = np.unique(points_pool[:, -1], return_counts=True)
                result = points[counts.argmax()]
            elif how == 'cl':
                wishart_algo = wishart.Wishart(w_neighbors, w_eps)
                clusters = wishart_algo.fit_predict(points_pool)
                cluster_labels, cluster_sizes = np.unique(clusters[clusters != 0], return_counts=True)

                if (cluster_labels.size > 0
                        and np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.8)) == 1):
                    biggest_cluster_center = points_pool[clusters == cluster_labels[cluster_sizes.argmax()]].mean()
                    result = biggest_cluster_center
                else:
                    result = np.nan
            '''cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

                if (cluster_labels.size > 0
                        and np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.8)) == 1):
                    biggest_cluster_center = points_pool[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                    result                 = biggest_cluster_center
                else:
                    result = np.nan'''

        return result

    def _fish(self, steps: int, eps: float, current_step: int) -> list:
        '''
        "Закидываем удочку". Из текущей точки закидываем все шаблоны вперед, насколько позволяет их длина, и сохраняем
        последние точки соответствующих обучающих векторов в массив points, которые затем аггрегируем в "общий" список
        forecasted_points.
        '''

        forecasted_points = []

        # длина удочки = максимальное расстояние между соседними зубчиками в шаблоне
        for i in range(min(self._max_template_spread, steps - current_step)):

            # вектора, последняя точка которых "висит в воздухе"
            last_vectors = (
                self._time_series[:self._original_size + current_step]
                [np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]
                 + (self._template_shapes[:, -1] - 1).reshape(-1, 1)]
            )

            # отсеиваем "недостаточно длинные" шаблоны
            length_mask = self._template_shapes[:, -1] > i

            # убираем вектора, в которые попадают непрогнозируемые точки
            non_nan_mask = ~np.isnan(last_vectors).any(axis=1)

            total_mask = length_mask & non_nan_mask

            last_vectors    = last_vectors[total_mask]
            distance_matrix = self._calc_distance_matrix(last_vectors, total_mask, steps)

            points = self._training_vectors[total_mask][distance_matrix < eps, -1]
            forecasted_points.append(points)

        return forecasted_points

    def push_one(self, steps, eps, w_eps, w_neighbors, real_points, phi=1e9):
        '''Предсказать ряд на steps вперёд, используя только начальный ряд'''
        self._training_vectors     = np.hstack([self._training_vectors,
                                                np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)])
        self._time_series          = np.resize(self._time_series, self._original_size + steps)
        self._time_series[-steps:] = np.nan
        point_pools = [[] for _ in range(steps)]
        for step in range(steps):
            for i in range(self._templates.shape[0]):
                pattern = (self._templates[i] * -1 + self._original_size + step)[::-1]
                if pattern[0] < 0:
                    continue
                block = self._time_series[pattern]
                if np.any(np.isnan(block[:-1])):
                    continue
                blocks = self._training_vectors[i]
                distance = np.sqrt(((blocks[:, :-1] - block[:-1]) ** 2).sum(axis=1))
                for block in blocks[distance < eps]:
                    point_pools[step].append(block)
            point_pools[step] = np.array(point_pools[step])
            predict = self._freeze_point(point_pools[step], self.type_algo, w_eps, w_neighbors)
            if predict != np.nan and abs(predict - real_points[step]) < phi:
                self._time_series[self._original_size + step] = predict
                for i in range(self._templates.shape[0]):
                    pattern = (self._templates[i] * -1 + self._original_size + step)[::-1]
                    if pattern[0] < 0:
                        continue
                    block = self._time_series[pattern]
                    if np.any(np.isnan(block)):
                        continue
                    self._training_vectors[i][pattern[0]] = block

        return self._time_series[-steps:]