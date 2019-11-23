import jieba
import pandas as pd
import pickle
import numpy as np


def test_data():
    test_docs = pd.read_csv('../data/test_docs.csv')
    title_seg = []
    test_docs[test_docs['doc_title'].isnull()] = '无'
    for target in test_docs['doc_title']:
        seg_list = jieba.lcut(target, cut_all=False)
        title_seg.append(' '.join(seg_list))

    print(title_seg)
    with open('./title_seg.o', 'wb') as p:
        pickle.dump(title_seg, p)


if __name__ == '__main__':
    with open('./title_seg.o', 'rb') as r:
        title_seg = pickle.load(r)
    print(title_seg)
