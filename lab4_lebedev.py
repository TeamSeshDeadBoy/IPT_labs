import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class Hardening:
    """
    Класс реализует функцию цикла отжига
    """
    
    def __init__(self, funct, tmax, tmin, alpha=0.89):
        # Гиперпараметры
        self.funct = funct
        self.tmax = tmax
        self.t = tmax
        self.tmin = tmin
        self.alpha = alpha
        self.S0 = np.random.uniform(low=0,high=500)
        self.cachedS0 = []
        
    def get_energylvl(self, val):
        """
        Функция получает "уровень энергии" заданной функциии при поданном x
        Args:
            val (float): значение x
        Returns:
            float: значение F(x) 
        """
        
        return self.funct(val)
    
    def calculate(self):
        """
        Полный цикл отжига
        """
        
        t = 0
        # Цикл отжига
        while self.t >= self.tmin:
            # Получаем новое значение x
            Si = self.S0 + np.random.uniform(low=-0.055,high=0.055)*self.t
            # Значение функции при x
            E_so = self.get_energylvl(self.S0)
            if (0<=Si and Si<=100):
                E_si = self.get_energylvl(Si)
                if E_si-E_so<0:
                    # Меняем локальный минимум
                    self.S0 = Si
                    self.cachedS0.append(self.S0)

                else:
                    # При попадании вероятности обновлячем локальный минимум
                    p=np.exp(-(E_si-E_so)/self.t)
                    r=np.random.uniform(low=0,high=1)
                    if r<p:
                        self.S0=Si
                        self.cachedS0.append(self.S0)

            self.t = self.tmax / np.log(1 + t)
            print(self.t)
            t += 1


def function(x):
    y=x**3-60*x**2-4*x+6
    return y

alg = Hardening(function, 1800, 180)
alg.calculate()
print(alg.S0)


x=[i/10 for i in range(1000)]
y=[0 for i in range(1000)]
for i in range(1000):
    y[i]=function(x[i])
    
st.header("Алгоритм Отжига")
st.write("График заданной функции")
fig2 = plt.figure(figsize=(10, 4))
plt.plot(x,y)
st.pyplot(plt)

st.write("Аппроксимация минимума функции")

fig = plt.figure(figsize=(10, 4))
cache = [alg.cachedS0[i] for i in range(0,len(alg.cachedS0),100)]
cache2 = [function(i) for i in cache]
plt.plot(x,y)
plt.scatter(cache, cache2,c="red")
st.pyplot(plt)
            
        