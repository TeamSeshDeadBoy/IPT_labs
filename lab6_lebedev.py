import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder


class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        # Размер матриц весов для слоев нейронной сети
        self.sizes = sizes
        # Кол-во эпох
        self.epochs = epochs
        # Гиперпараметр значимости градиента (скорость схождения градиента \ обучения)
        self.l_rate = l_rate

        # Кэширование параметров
        self.params = self.initialization()


    def sigmoid(self, x, derivative=False):
        """Функция Сигмоиды с реализацией через производную для обратного прохода
        Args:
            x (np.array[]): массив чисел на вход функции
            derivative (bool, optional): флаг обратного прохода. Defaults to False.

        Returns:
            np.array[]: массив чисел выхода из сигмоидной функции
        """
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        """Функция софтмакса приводит показания к вероятностям для категориальных задач предсказания

        Args:
            x (np.array[float]): массив предсказаний
            derivative (bool, optional): флаг обратного прохода. Defaults to False.

        Returns:
            np.array[float]: выход из софтмакс функции
        """
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        """Инициализация параметров нейронной сети

        Returns:
            dict{}: Словарь с матрицами весов слоев нейронной сети
        """
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        """Функция имитирует проход по нейронной сети. От исходных данных к предсказанию модели

        Args:
            x_train (np.array[][]): Матрица исходных данных, поданных в нейронную сеть

        Returns:
            np.array[]: Активации нейронов последнего слоя. Предсказания модели.
        """
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"].astype('float32'), params['A0'].astype('float32'))
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            if type(x) != type(' '):
                output = self.forward_pass(x)
                pred = np.argmax(output)
                predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        epochs_bar = st.progress(0)
        for iteration in range(self.epochs):
            epochs_bar.progress((iteration + 1)/self.epochs, "Обучение модели:")
            cnt = 1
            max_epoch = x_train.shape[0]
            epoch_bar = st.progress(0)
            for x,y in zip(x_train, y_train):
                epoch_bar.progress(cnt/max_epoch, f"Обучение эпохи {iteration + 1}")
                cnt += 1
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            accuracy = self.compute_accuracy(x_val, y_val)
            st.write('Эпоха: {0}, Время на эпоху: {1:.2f}s, Метрика аккуратности: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
        epochs_bar.progress(1)
    
    
###-----------DATA-PREPROCESSING--------------
@st.cache_data
def download_data():
    """Скачивает и кэширует датасет MNIST с рукописными цифрами
    Returns:
        np.array[][], np.array[][]: x и y датасета
    """
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    return x, y

# Helper function
def show_images(image, num_row=2, num_col=5):
    """Выводит на streamlit поданные в функцию картинки через matplotlib.pyplot

    Args:
        image (arr[matplotlib.figure]): Массив картинок для вывдоа
        num_row (int, optional): кол-во рядов. Defaults to 2.
        num_col (int, optional): кол-во колонок. Defaults to 5.
    """
    # plot images
    image_size = int(np.sqrt(image.shape[-1]))
    image = np.reshape(image, (image.shape[0], image_size, image_size))
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    
def show_image(image):
    """Выводит на streamlit поданные в функцию картинки через matplotlib.pyplot

    Args:
        image (arr[matplotlib.figure]): Массив картинок для вывдоа
        num_row (int, optional): кол-во рядов. Defaults to 2.
        num_col (int, optional): кол-во колонок. Defaults to 5.
    """
    # plot images
    image_size = int(np.sqrt(image.shape))
    image = np.reshape(image,  (image_size, image_size))
    fig, ax = plt.subplots(1, 1, figsize=(1.5*1,2*1))
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    
@st.cache_resource
def cache_load(x_train, y_train, x_val, y_val ,epochs=10):
    dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10], epochs=epochs)
    dnn.train(np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val))
    return dnn

def main():
    ### Скачиваем данные, препроцессинг
    x, y = download_data()
    # x = pd.to_numeric(x)
    # y = pd.to_numeric(y)
    x = (x/255).astype('float32')
    st.write(y.shape)
    y = pd.get_dummies(y, dtype='float32')
    st.write(y.shape)

    ### Сплитуем данные
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

    ### Streamlit вывод
    st.header("Создание Нейронной сети (2 скрытых слоя) для предсказания рукописных цифр с датасета MNIST")
    st.write("Исходные данные")
    st.write(f"Размер обучающей выборки (train): {x_train.shape} {y_train.shape}")
    st.write(f"Размер валидирующей выборки (test): {x_val.shape} {y_val.shape}")
    st.write("Примеры изображений:")
    show_images(np.array(x_train))

    st.dataframe(np.array(x_train))
    st.write("Обучение модели:")
    dnn = cache_load(np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), 10)
    st.write("Обучение завершено.")
    checker = st.button("Проверить модель на случайном числе")
    if checker:
        random_number_1 = np.random.randint(1,1000)
        random_number_2 = np.random.randint(1,1000)
        random_number_3 = np.random.randint(1,1000)
        random_number_4 = np.random.randint(1,1000)
        random_picture_1 = x_val.iloc[random_number_1]
        random_picture_2 = x_val.iloc[random_number_2]
        random_picture_3 = x_val.iloc[random_number_3]
        random_picture_4 = x_val.iloc[random_number_4]
        show_images(np.array([random_picture_1, random_picture_2, random_picture_3, random_picture_4]), 2, 2)
        st.write("1 Фото - Модель предсказала:", np.argmax(dnn.forward_pass(random_picture_1)), "Настоящее число: ", np.argmax(y_val.iloc[random_number_1]))
        st.write("2 Фото - Модель предсказала:", np.argmax(dnn.forward_pass(random_picture_2)), "Настоящее число: ", np.argmax(y_val.iloc[random_number_2]))
        st.write("3 Фото - Модель предсказала:", np.argmax(dnn.forward_pass(random_picture_3)), "Настоящее число: ", np.argmax(y_val.iloc[random_number_3]))
        st.write("4 Фото - Модель предсказала:", np.argmax(dnn.forward_pass(random_picture_4)), "Настоящее число: ", np.argmax(y_val.iloc[random_number_4]))

if __name__=='__main__':
    main()