import streamlit as st

st.text_input("Введите строку для кодирования", key="st_string")

# Класс дерева для создания бинарного дерева
class NodeTree(object):
    def __init__(self,char=None, weight=None, left=None, right=None):
        self.char = char
        self.weight = weight
        self.left = left
        self.right = right


def create_dict(input_string):
    """
    Возращает отсортированный словарь уникальных символов в строке с
    количеством их вхождений в строку
    Args:
        :param input_string (string): входная строка
    """
    input_length = len(input_string)
    chars_dict = dict(zip(set(input_string), [0]*input_length))
    for i in list(input_string):
        chars_dict[i] += 1
    # Сортируем словарь по значениям
    chars_dict =  dict(sorted(chars_dict.items(), key=lambda x: x[1], reverse=True))
    return chars_dict


def create_head(dictionary):
    """
    Создает головную ноду дерева
    Args:
        dict (_type_): Входной словарь с символами и их вхождениями
    """
    left = NodeTree(list(dictionary.keys())[-1], dictionary[list(dictionary.keys())[-1]])
    dictionary.pop(list(dictionary.keys())[-1])
    right = NodeTree(list(dictionary.keys())[-1], dictionary[list(dictionary.keys())[-1]])
    dictionary.pop(list(dictionary.keys())[-1])
    head = NodeTree("",left.weight+right.weight, left, right)
    return head, dictionary
    
        

def create_tree(node, dict):
    """
    Создает бинарное дерево для кодирования методом Хаффмана.
    Args:
        node (NodeTree): рекурсивно переданная нода.
        dict (str): словарь с символами.
    Returns:
        NodeTree: головная нода созданного дерева
    """
    if len(list(dict.keys())) == 0:
        return node
    else:
        if node.weight > dict[list(dict.keys())[-1]]:
            right = NodeTree(list(dict.keys())[-1], dict[list(dict.keys())[-1]])
            dict.pop(list(dict.keys())[-1])
            node = NodeTree("", node.weight + right.weight, node, right)
        else:
            left = NodeTree(list(dict.keys())[-1], dict[list(dict.keys())[-1]])
            dict.pop(list(dict.keys())[-1])
            node = NodeTree("", node.weight + left.weight, left, node)
        return create_tree(node, dict)


def BFS(node, search_char, path=""):
    """
    BFS по дереву для сбора сроки кодирования символа
    Args:
        node (NodeTree): рекурсивно переданная нода
        search_char (str): символ для поиска
        path (str, optional): строка для сбора бинарного кода. Defaults to "".

    Returns:
        str: бинарный код закодированного символа
    """
    if not node:
        return ""
    if node.char == search_char:
        return path
    res = BFS(node.right,search_char, path+"0") + BFS(node.left,search_char, path+"1")
    return res

def decode(string, codes):
    """
    Превращает строку в код с помощбю словаря
    Args:
        string (str): входная строка
        codes (dict): словарь с кодами символов
    Returns:
        str: закодированная строка
    """
    output = ""
    for i in list(string):
        output += codes[i]
    return output
    
if st.button("Закодировать"):
    frequencies_dict = create_dict(st.session_state.st_string)
    st.write("Частота появления символов")
    st.dataframe(frequencies_dict)
    (tree_head, frequencies_dict) = create_head(frequencies_dict)
    tree_head = create_tree(tree_head, frequencies_dict)
    encoded_dict = {}
    for i in list(create_dict(st.session_state.st_string).keys()):
        encoded_dict[i] = BFS(tree_head, i)
    st.write("Закодированные символы")
    st.dataframe(encoded_dict)
    decoded_str = decode(st.session_state.st_string, encoded_dict)
    st.write("Закодированная строка", int(decoded_str))
    st.write("Длина закодированной строки:", len(decoded_str))