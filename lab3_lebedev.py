import streamlit as st
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

class graph:
    """
    Класс создает случайный граф с заданным кол-вом вершин и высчитывает расстояния между ними
    """
    def __init__(self, num=10):
        self.num_dots = num
        self.coords = np.random.rand(self.num_dots, 2)
        self.distances = spatial.distance.cdist(self.coords, self.coords, metric='euclidean')
       

class AntColony:
    """Класс представляет собой реализацию работы Муравьиного Алгоритма
    """
    def __init__(self, num_ants, coords, distances, epochs=10, a=0.7, b=1.5, rho=0.45, Q=120):
        self.num_ants = num_ants
        self.coords_list = coords
        self.distances_list = distances
        self.num_cities = len(self.coords_list)
        self.epochs = epochs
        # Коэффициент запаха муравьев
        self.a = a
        # Коэффициент расстояния
        self.b = b
        # Коэффициент интенсивности исспарения
        self.rho = rho
        # Коэффициент кол-ва выпускаемого феромона
        self.Q = Q
        self.BEST_DIST = float("inf")
        self.BEST_PATH = None
        
        
    def count_paths(self):
        # Начальный уровень феромона в вершинах
        ph_start = self.Q/self.num_cities
        # Матрица уровней феромона
        ph_matrix = np.ones((self.num_cities, self.num_cities)) * ph_start
        # Матрицы отслеживания путей и весов
        ant_route = np.zeros((self.num_ants, self.num_cities))
        ant_distances = np.zeros(self.num_ants)
        self.ant_best_dist = np.zeros(self.epochs)
        self.ant_avg_dist = np.zeros(self.epochs)
        
        # Streamlit прогресс бар
        progress_bar = st.progress(0)
        
        for epoch in range(self.epochs):
            progress_bar.progress(epoch/self.epochs)
            # Обновляем матрицы путей и весов
            ant_route.fill(0)
            ant_distances.fill(0)
            for ant in range(self.num_ants):
                #print("Ant #", ant)
                # Начальное положение муравьев равномерное
                ant_route[ant, 0] = np.random.randint(0, self.num_cities-1)
                
                # Начальное положение муравьев константа
                # ant_route[ant, 0] = 1
                
                # Обходим граф муравьем:
                for step in range(1, self.num_cities):
                    #print("Step", step)
                    # Получаем город отправления муравья
                    prev_city = int(ant_route[ant, step-1])
                    P = (ph_matrix[prev_city] ** self.a) * (1/self.distances_list[prev_city] ** self.b)
                    
                    # Обнуляем вероятности муравья пойти в уже посещенные вершины
                    for i in range(step):
                        P[int(ant_route[ant, i])] = 0
                    P = P / np.sum(P)
                    
                    # Случайно выбираем город для муравья
                    #print("Choosing random city")
                    flag = True
                    while flag:
                        rand = np.random.random()
                        for p, to in zip(P, list(range(num_cities))):
                            #print("p :", p, "rand :", rand)
                            if p >= rand:
                                # Записываем город в путь муравья
                                ant_route[ant, step] = to
                                flag = False
                                break
                # Записываем длину маршрута муравья
                for i in range(self.num_cities):
                    city_from = int(ant_route[ant, i-1])
                    city_to = int(ant_route[ant, i])
                    ant_distances[ant] += self.distances_list[city_from, city_to]
                
                # Сравниваем ее с минимальными показателями для колонии
                if ant_distances[ant] < self.BEST_DIST:
                    self.BEST_DIST = ant_distances[ant]
                    self.BEST_PATH = ant_route[ant]
                    
            # Высыхание феромона
            ph_matrix = ph_matrix * (1-self.rho)
                
            # Обновляем феромон:
            for ant in range(self.num_ants):
                for step in range(self.num_cities):
                    city_to = int(ant_route[ant, step])
                    city_from = int(ant_route[ant, step-1])
                    # Обновление феромона по формуле
                    ph_matrix[city_from, city_to] = ph_matrix[city_from, city_to] + (self.Q / ant_distances[ant])
                    # Обратные пути
                    ph_matrix[city_to, city_from] = ph_matrix[city_from, city_to]
            
            self.ant_best_dist[epoch] = self.BEST_DIST
            self.ant_avg_dist[epoch] = np.average(ant_distances)
                
        # Обновляем прогресс бар
        # progress_bar.progress(1.0, text="Обход колонии завершен")
        

st.header("Муравьиный Алгоритм в задаче коммивояжёра")

num_cities=st.slider(min_value=5, max_value=25, label="Кол-во городов", value=8)
num_epochs=st.slider(min_value=num_cities * 30, max_value=num_cities * 100, label="Кол-во Эпох обучения", value=num_cities * 90)
a=st.slider(min_value=0.1, max_value=2.0, label="Коэффициент A (жадность)", value=0.3)
b=st.slider(min_value=0.1, max_value=2.0, label="Коэффициент B (Феромон)", value=1.1)
cities = graph(num_cities)
fig = plt.figure(figsize=(10, 4))
plt.scatter(cities.coords[:,0], cities.coords[:,1],c="red")
st.pyplot(fig)


if st.button("Начать алгоритм поиска"):
    colony = AntColony(num_ants=20, coords=cities.coords, distances=cities.distances, epochs=num_epochs, a=a, b=b, rho=0.45, Q=120)
    colony.count_paths()
        
    x = list(range(colony.epochs))
    fig = plt.figure(figsize=(10, 4))
    plt.plot(x, colony.ant_best_dist,c="red")
    # plt.plot(x, colony.ant_avg_dist, c="blue")
    st.pyplot(plt)
        
    st.write(colony.BEST_DIST)
    st.write(colony.BEST_PATH)
    path = []
    for i in colony.BEST_PATH:
        path.append(list(cities.coords[int(i)]))
            
    path = np.array(path)
    fig = plt.figure(figsize=(10, 4))
    plt.scatter(cities.coords[:,0], cities.coords[:,1],c="red")
    plt.plot(path[:,0], path[:,1], c="green")
    st.pyplot(plt)