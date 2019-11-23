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


def test_querys():
    test_querys = pd.read_csv('../data/test_querys.csv')
    querys_seg = []
    test_querys[test_querys['query'].isnull()] = '无'
    for target in test_querys['query']:
        seg_list = jieba.lcut(target, cut_all=False)
        querys_seg.append(' '.join(seg_list))

    print(querys_seg)
    with open('./querys_seg.o', 'wb') as p:
        pickle.dump(querys_seg, p)


if __name__ == '__main__':
    # with open('./title_seg.o', 'rb') as r:
    #     title_seg = pickle.load(r)
    # print(title_seg)

    # test_querys()

    with open('./querys_seg.o', 'rb') as r:
        title_seg = pickle.load(r)
    print(title_seg)
