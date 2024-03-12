import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import streamlit as st

import numpy as np
import cv2
import pickle
import sys

class Weights(object):
    def __init__(self, array):
        self.weights = array

def Hebb():
    filenames = ['o.jpg', 'x.jpg', 'x2.png']
    with open('weights/x_o.pkl', 'rb') as input:
        dump = pickle.load(input)
        print("Loading Weight Files")
        weights = dump.weights

    for file in filenames:
        img = cv2.imread('test/'+file, cv2.IMREAD_GRAYSCALE)
        fig = plt.figure(figsize=[10,10])
        plt.imshow(img)
        st.pyplot(fig)
        img = cv2.resize(img, (500, 500))
        img_w = []
        for i in img:
            for j in i:
                if j>128:
                    img_w.append(-1)
                else:
                    img_w.append(1)

        f_y = 0

        for index, w in enumerate(weights):
            f_y += w * img_w[index]

        if f_y >= 0:
            st.write("На фотографии: O")
        else:
            st.write("На фотографии: X")
        
        

class Kohonen():
    def __init__(self, n_dims: list[int]=[10,10]) -> None:
        """Класс реализует работу алгоритма Кохонена. Нейронная сеть с самоокластеризацией цветов

        Args:
            n_dims (list[int], optional): Рамзеры сетки нейронов. Defaults to [10,10].
        """
        # Параметры нейронной сети
        raw_data = np.random.randint(0, 255, (3, 100))
        self.network_dimensions = np.array(n_dims)
        self.n_iterations = 10000
        self.init_learning_rate = 0.01

        # Вспомогательные параметры
        self.normalise_data = True
        self.normalise_by_column = False
        self.m = raw_data.shape[0]
        self.n = raw_data.shape[1]

        # Радиус соседей
        self.init_radius = max(self.network_dimensions[0], self.network_dimensions[1]) / 2
        # Параметр уменьшения радиуса
        self.time_constant = self.n_iterations / np.log(self.init_radius)

        # Нормирование данных
        self.data = raw_data
        if self.normalise_data:
            if self.normalise_by_column:
                col_maxes = raw_data.max(axis=0)
                self.data = raw_data / col_maxes[np.newaxis, :]
            else:
                self.data = raw_data / self.data.max()
                
        # Инициальзация случайных весов
        self.net = np.random.random((self.network_dimensions[0], self.network_dimensions[1], self.m))
        
    def get_data(self) -> np.ndarray:
        """Геттер для данных

        Returns:
            np.ndarray: Массив входных данных модели
        """
        return self.data
                
    def find_bmu(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            Находит ближайший нейрон, подходящий к поданному вектору цвета.
        """
        bmu_idx = np.array([0, 0])
        min_dist = np.iinfo(int).max
        
        # Итерируемся по нейронам и ищем подходящий с помощью
        for x in range(self.net.shape[0]):
            for y in range(self.net.shape[1]):
                w = self.net[x, y, :].reshape(self.m, 1)
                sq_dist = np.sum((w - t) ** 2)
                sq_dist = np.sqrt(sq_dist)
                if sq_dist < min_dist:
                    min_dist = sq_dist # dist
                    bmu_idx = np.array([x, y]) # id
        
        # Выводим нейрон и его индекс
        bmu = self.net[bmu_idx[0], bmu_idx[1], :].reshape(self.m, 1)
        return (bmu, bmu_idx)
    
    
    def decay_radius(self, i: int) -> float:
        """Уменьшает радиус области соседей
        Args:
            i (int): индекс итерации

        Returns:
            float: новый радиус
        """
        return self.init_radius * np.exp(-i / self.time_constant)


    def decay_learning_rate(self, i: int) -> float:
        """Уменьшает параметр Learning Rate для логарифмичности обучения

        Args:
            i (int): Индекс итерации

        Returns:
            float: Новый параметр Learning Rate
        """
        return self.init_learning_rate * np.exp(-i / self.n_iterations)


    def calculate_influence(self, distance: float, radius: float) -> float:
        """Функция высчитывает влияние

        Args:
            distance (float): _description_
            radius (float): _description_

        Returns:
            float: _description_
        """
        return np.exp(-distance / (2* (radius**2)))
    
    
    def iterate(self, epochs: int) -> None:
        """Процесс обучения с помощью итерации

        Args:
            epochs (int): Кол-во эпох итерации
        """
        for i in range(epochs):
            # Случайно выбранный экземпляр из обучающей выборки
            t = self.data[:, np.random.randint(0, self.n)].reshape(np.array([self.m, 1]))
            
            # Находим ближайший нейрон
            bmu, bmu_idx = self.find_bmu(t)
            
            # Уменьшаем параметры
            r = self.decay_radius(i)
            l = self.decay_learning_rate(i)
            
            # Обновляем вектора весов
            # Двигаем соседей ближе
            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    w = self.net[x, y, :].reshape(self.m, 1)
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                    w_dist = np.sqrt(w_dist)
                    
                    if w_dist <= r:
                        # Получаем метрику близости нейрона к вектору
                        influence = self.calculate_influence(w_dist, r)
                        
                        # Меняем веса нейрона
                        new_w = w + (l * influence * (t - w))
                        self.net[x, y, :] = new_w.reshape(1, 3)
                    
                    
    def show(self, iter: int) -> None:
        """Функция выводит веса нейронной сети в виде plt графика

        Args:
            iter (int): Кол-во итераций. Вставляется в описание графика.ы
        """
        fig = plt.figure()

        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(0, self.net.shape[0]+1)
        ax.set_ylim(0, self.net.shape[1] + 1)
        ax.set_title('Self-Organising Map after %d iterations' % iter)

        # plot
        for x in range(1, self.net.shape[0] + 1):
            for y in range(1, self.net.shape[1] + 1):
                ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                            facecolor=self.net[x-1,y-1,:],
                            edgecolor='none'))

        st.pyplot(fig)
    
def show_colors(colors):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    colors = np.array(colors).T
    
    for ind in range(len(colors)):
        ax.add_patch(patches.Rectangle((ind-0.5, 0.5), 1, 1,
                            facecolor=colors[ind],
                            edgecolor='none'))
    st.pyplot(fig)
                        

st.header("Обучение ИНС без учителя (с подкреплением)")                
st.header('Алгоритм Хэбба для классификации изображений')
Hebb()

# Основная работа streamlit дэшборда                        
st.header("Алгоритм Кохонена для самоклассификации цветов")
n_dims = st.slider("Размер сетки нейронов:", min_value=5, max_value=20)

dnn = Kohonen([n_dims, n_dims])
st.dataframe(dnn.get_data())

show_colors(dnn.get_data())

dnn.show(0)
dnn.iterate(100)
dnn.show(100)
dnn.iterate(1000)
dnn.show(1000)
dnn.iterate(10000)
dnn.show(10000)

