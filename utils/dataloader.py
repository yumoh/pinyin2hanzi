from itertools import chain
from collections import Counter
import numpy as np


class DataLoader:
    def __init__(self, data_path: str = "./data/zh.tsv"):
        self.data_path = data_path

        # 原始的拼音list,原始的字符list
        self.data_pinyin, self.data_char = self._load_to_py_to_char()
        self.max_data_len = len(self.data_pinyin)
        # 拼音序列的最大长度,字符序列的最大长度
        self.max_pinyin_seq_len = max(map(len, self.data_pinyin)) + 2
        self.max_char_seq_len = max(map(len, self.data_char)) + 2
        # index和字符之间的相互转化
        self.char_to_index, self.pinyin_to_index = self._compute_index(self.data_pinyin, self.data_char)
        self.index_to_char = {v: k for k, v in self.char_to_index.items()}
        self.index_to_pinyin = {v: k for k, v in self.pinyin_to_index.items()}

        # 单词的个数
        self.char_numbers = len(self.char_to_index) + 1
        self.pinyin_numbers = len(self.pinyin_to_index) + 1
        # 拼音的array
        self.data_pinyin_array = np.stack([self._pad_to_array([self.pinyin_to_index[i] for i in s])
                                           for s in self.data_pinyin])
        # series length
        self.data_pinyin_seq_len_array = np.stack([len(s) for s in self.data_pinyin])
        # 字符的array
        self.data_char_array = np.stack([self._pad_to_array([self.char_to_index[i] for i in s])
                                         for s in self.data_char])
        # series length
        self.data_char_seq_len_array = np.stack([len(s) for s in self.data_char])

    def __repr__(self):
        return f"""data:{self.data_path}
            series size:{self.max_data_len}
            input data max length:{self.max_pinyin_seq_len} words:{self.pinyin_numbers}
            target data max length:{self.max_char_seq_len} words:{self.char_numbers}
            """

    def _load_to_py_to_char(self):
        data_py = []
        data_char = []
        with open(self.data_path, encoding='utf8') as fp:
            line = fp.readline()
            while line:
                line = line.split('\t')
                data_py.append(['<pad>'] + line[1].strip().split(' '))
                data_char.append(['<pad>'] + line[2].strip().split(' '))
                line = fp.readline()

        return data_py, data_char

    def _compute_index(self, data_py, data_char):
        py_count = Counter(chain(*data_py))
        char_count = Counter(chain(*data_char))

        char_to_index = dict(zip(char_count, range(1, 1 + len(char_count))))
        py_to_index = dict(zip(py_count, range(1, 1 + len(py_count))))

        return char_to_index, py_to_index

    def _pad_to_array(self, series, max_length=50):
        a = np.array(series + [0] * (max_length - len(series)), dtype=np.long)
        return a

    def array_to_pinyin(self, series: np.array, as_list: bool = False):
        """
        转换成拼音
        :param series: 序列
        :param as_list: 是否要作为list返回
        :return: result
        """
        result_list = [self.index_to_pinyin[int(i)] for i in series if i > 0]
        if as_list:
            return result_list
        return ' '.join(result_list)

    def array_to_char(self, series: np.array, as_list: bool = False):
        """
        转换成汉字
        :param series: 序列
        :param as_list: 是否要作为list返回
        :return: result
        """
        result_list = [self.index_to_char[int(i)] for i in series if i > 0]
        if as_list:
            return result_list
        return ' '.join(result_list)

    def gen_batch(self, size=128, loops=-1):
        """
        迭代返回数据用于训练或测试
        :param size: batch size
        :param loops: 训练loops
        :return: yield list
        """
        loop = 0
        while loops < 0 or loop <= loops:
            indexes = np.random.randint(0, self.max_data_len - 1, size=size)
            r = [self.data_pinyin_array[indexes], self.data_pinyin_seq_len_array[indexes],
                 self.data_char_array[indexes], self.data_char_seq_len_array[indexes]]
            yield r
            loop += 1
