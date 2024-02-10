import streamlit as st
import pandas as pd
import numpy as np

st.text_input("Введите строку", key="input")

# Кодируем
def decode(input):
    inp_len = len(input)
    # Словарь с вероятностями
    probs_dict = dict(zip(sorted(set(input)), [0] * inp_len))
    for i in list(input):
        probs_dict[i] += 1/inp_len
    st.write("Вероятности")
    st.dataframe(probs_dict)

    # Суммируем вероятности чтобы получить верхние границы отрезков вероятности
    probs_summated = [0]
    for i in probs_dict:
        probs_summated.append(probs_dict[i] + probs_summated[-1])
    summated_dict = dict(zip(sorted(set(input)), probs_summated[1:]))
    st.write(summated_dict)

    # Алгоритм кодирования
    start = 0
    end = 1
    log = []
    for i in list(input):
        # Зменяем границы отрезков
        length = end - start
        end = start + length * summated_dict[i]
        start = end - length * probs_dict[i]
        log.append([i, start, end])

    st.write("Выполняем арифметическое кодирование")
    st.dataframe(log)
    code = (end - start) / 2 + start
    st.write("Закодированное значение", code)
    return code, summated_dict, probs_dict
    

# Декодируем
def encode(code, summ_dict, probs_dict, lenn):
    # Алгоритм декодирования
    start = 0
    end = 1
    length = end - start
    log = []
    target = code
    decoded_word = ''
    for i in range(lenn):
        # Определяем по таргету какая буква закодирована
        for let, prob in summ_dict.items():
            if prob >= target:
                letter = let
                break
        # Изменяем границы отрезка
        end = start + length * summ_dict[letter]
        start = end - length * probs_dict[letter]
        length = end - start
        # Считаем новый таргет
        target = (code - start) / length
        # Записываем букву
        decoded_word += letter[0]
        log.append([letter, target, start, end])
    st.write("Выполняем арифметическое кодирование")
    st.dataframe(log)
    st.write("Декодированное слово", decoded_word)
    return decoded_word

if st.button("Закодировать"):
    code, probs_s, probs = decode(st.session_state.input)
    encode(code, probs_s, probs, len(st.session_state.input))
