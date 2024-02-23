import numpy as np
from scipy import spatial
import streamlit as st
import matplotlib.pyplot as plt

class graph:
    """
    Класс создает случайный граф с заданным кол-вом вершин и высчитывает расстояния между ними
    """
    def __init__(self, num=10):
        self.num_dots = num
        self.coords = np.random.rand(self.num_dots, 2)
        self.distances = spatial.distance.cdist(self.coords, self.coords, metric='euclidean')


class Henetics():
    """Класс имитирует популяцию с хромомсомами и генами, обозначающими города
    """
    def __init__(self, population_size, chromosome_length, distances):
        self.population = self.create_population(population_size, chromosome_length)
        self.targets = self.evaluate_population(self.population, distances)

    def create_population(self, size, chr_len):
        """Создает случайную популяцию и хромосомы
        Args:
            size (int): размер популяции
            chr_len (int): длина хромосомы
        """
        population = list()
        for _ in range(size):
            chromosomes = list(range(1,chr_len))
            # Принимаем стартовый город за 0 (игнорируем его)
            np.random.shuffle(chromosomes)
            chromosomes = [0] + chromosomes + [0]
            population.append(chromosomes)
        return population

    def cache_init(self, c_population, distances):
        """Функция позволяет использорвать кжшированную популяцию как новый обьект популяции

        Args:
            c_population (arr[][]): кэшированный массив хромосом
            distances (arr[][]): дистаниции между городами для эвалуации
        """
        self.population = c_population
        self.targets = self.evaluate_population(self.population, distances)

    def add_gene(self, gene, distances):
        """Функция добавляет и эвалуирует новый (мутированный извне) ген
        Args:
            gene (arr[]): ген
            distances (arr[][]): дистанции мсежду городами для эвалуации
        """
        self.population.append(gene)
        self.targets = self.evaluate_population(self.population, distances)

    def remove_last(self):
        """Функция удаляет последнюю хросмосому из популяции
        """
        self.population.pop()
        self.targets.pop()

    def evaluate_population(self, population, distances):
        """Подсчитываем таргет для каждой особи(хромосомы особи) - считаем длину маршрута
        Args:
            population (arr[][]): массив популяции особей
            distances (arr[][]): матрица расстояний между городами

        Returns:
            arr[]: массив таргетов особей
        """
        targets = []
        for chromosome in population:
            length = 0
            for i in range(1,len(chromosome)):
                city_start = chromosome[i-1]
                city_end = chromosome[i]
                length += distances[city_start, city_end]
            targets.append(length)
        return targets


class GA():
    """Класс реализует генетический алгоритм для решения задачи коммивояжёра
    """
    def __init__(self,num_cities, population_size):
        # Создаем граф городов размера num_cities
        self.Cities = graph(num_cities)
        # Создаем популяцию размера population_size, с хромосомами определяющими маршрут по графу
        self.Population = Henetics(population_size, num_cities, self.Cities.distances)


    def mutate_gene(self, gene):
        """Мутирует ген одной случайной перестановкой
        Args:
            gene (arr[]): ген

        Returns:
            arr[]: Мутированный ген
        """
        rv1 = np.random.randint(1,len(self.Population.population[0]) - 2)
        rv2 = np.random.randint(1,len(self.Population.population[0]) - 2)
        while rv2 == rv1:
            rv2 = np.random.randint(1,len(self.Population.population[0]) - 2)
        temp = gene[rv2]
        gene[rv2] = gene[rv1]
        gene[rv1] = temp
        return gene

    def iterate(self, generation_threshold, temperature):
        """Процесс итерирования (эволюции популяции)

        Args:
            generation_threshold (int): максимальное кол-во поколений популяции
            temperature (int): температура (влияет на вероятность принятия плохого решения для вариативности)
        """
        progress_bar = st.progress(0)
        gen = 0
        # Кэшируем для вывода
        self.MIN_PATH = []
        self.MIN_length = float("inf")
        self.AVG_CACHE = []
        while gen <= generation_threshold:
            progress_bar.progress(gen/generation_threshold, "Процесс итерации")
            # Скрещивание и мутации потомков популяции
            # Создаем новую пустую популяцию
            new_population = Henetics(0, 0, [[0, 0]])
            new_population.cache_init([], [])
            pop_length = self.Population.population
            self.AVG_CACHE.append(np.average(self.Population.targets))
            for i in range(len(pop_length)):
                p1 = self.Population.population[i]
                h1 = self.Population.targets[i]

                while True:
                    new_g = self.mutate_gene(p1)
                    new_population.add_gene(new_g, self.Cities.distances)

                    if new_population.targets[-1] > h1:
                        # Accepting the rejected children at
                        # a possible probability above threshold.
                        prob = pow(2.7,-1 * ((new_population.targets[-1] - h1) / temperature))
                        if prob < 0.75:
                            new_population.remove_last()
                        else:
                            break
                    else:
                        if new_population.targets[-1] < self.MIN_length:
                            self.MIN_length = new_population.targets[-1]
                            self.MIN_PATH = new_population.population[-1]
                        break
    
            temperature = 0.9 * temperature
            self.Population = new_population
            # print("Generation", gen)
            # print("GNOME     FITNESS VALUE")
    
            # for i in range(len(self.Population.population)):
            #     print(self.Population.population[i], self.Population.targets[i])
            gen += 1


st.header("Генетический алгоритм в задаче коммивояжера")
# st.write("Случайные города:")
# main = GA(5,10)
# main.iterate(300,100)


num_cities=st.slider(min_value=5, max_value=15, label="Кол-во городов", value=5)
num_genes=st.slider(min_value=5, max_value=40, label="Размер популяции", value=10)
num_generations=st.slider(min_value=20, max_value=500, label="Кол-во поколений", value=300)

main = GA(num_cities,num_genes)


cities = main.Cities
fig = plt.figure(figsize=(10, 4))
plt.scatter(cities.coords[:,0], cities.coords[:,1],c="red")
st.pyplot(fig)


if st.button("Начать алгоритм"):
    main.iterate(num_generations,100)
    
    st.write("Средняя длина пути в популяции")
    fig = plt.figure(figsize=(10, 4))
    plt.plot(main.AVG_CACHE)
    st.pyplot(plt)
    # x = list(range(colony.epochs))
    # fig = plt.figure(figsize=(10, 4))
    # plt.plot(x, colony.ant_best_dist,c="red")
    # # plt.plot(x, colony.ant_avg_dist, c="blue")
    # st.pyplot(plt)
        
    st.write("Минимальная длина пути: ", main.MIN_length)
    st.write("Минимальный путь: ", main.MIN_PATH)
    path = []
    for i in main.MIN_PATH:
        path.append(list(cities.coords[int(i)]))
            
    st.write("Путь")
    path = np.array(path)
    fig = plt.figure(figsize=(10, 4))
    plt.scatter(cities.coords[:,0], cities.coords[:,1],c="red")
    plt.plot(path[:,0], path[:,1], c="green")
    st.pyplot(plt)