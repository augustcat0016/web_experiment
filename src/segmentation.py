import jieba
import pandas as pd
import numpy as np


def test_data():
    test_docs = pd.read_csv('../data/test_docs.csv')
    title_seg = []
    for target in test_docs['doc_title']:
        seg_list = jieba.lcut(target, cut_all=False)
        print(seg_list)
        title_seg.append(' '.join(seg_list))

    print(title_seg)


if __name__ == '__main__':
    test_data()
