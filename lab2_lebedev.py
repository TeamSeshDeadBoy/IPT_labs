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


def create_tree(node, dicti, dangling_nodes=dict()):
    """
    Создает бинарное дерево для кодирования методом Хаффмана.
    Args:
        node (NodeTree): рекурсивно переданная нода.
        dicti (dict): словарь с символами.
        dangling_nodes (dict): словарь с неиспользованными нодами.
    Returns:
        NodeTree: головная нода созданного дерева
    """
    if len(list(dicti.keys())) == 1:
        return node
    else:
        # Берем два наименбших ключа в словаре
        smallest = list(dicti.keys())[-1]
        small = list(dicti.keys())[-2]

        # Если это новый лист - создаем его, если это уже созданная нода - достаем из кэша (словаря)
        headNode = NodeTree()
        if smallest in list(dangling_nodes.keys()):
            smallest_node = dangling_nodes[smallest]
            dangling_nodes.pop(smallest)
        else:
            smallest_node = NodeTree(smallest, dicti[smallest])

        if small in list(dangling_nodes.keys()):
            small_node = dangling_nodes[small]
            dangling_nodes.pop(small)
        else:
            small_node = NodeTree(small, dicti[small])

        # Распределяем их по ветвям
        headNode.right = smallest_node
        headNode.left = small_node

        # Добавляем новые элементы в словари
        dicti[smallest+small] = dicti[smallest] + dicti[small]
        dicti = dict(sorted(dicti.items(), key=lambda x: x[1], reverse=True))
        dangling_nodes[smallest+small] = headNode

        # удаляем использованные элементы
        dicti.pop(list(dicti.keys())[-1])
        dicti.pop(list(dicti.keys())[-1])
        return create_tree(headNode, dicti, dangling_nodes)


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
    tree_head = create_tree(NodeTree(), frequencies_dict)
    encoded_dict = {}
    for i in list(create_dict(st.session_state.st_string).keys()):
        encoded_dict[i] = BFS(tree_head, i)
    st.write("Закодированные символы")
    st.dataframe(encoded_dict)
    decoded_str = decode(st.session_state.st_string, encoded_dict)
    st.write("Закодированная строка", int(decoded_str))
    st.write("Длина закодированной строки:", len(decoded_str))