import jieba
import pandas as pd
import pickle


def test_data():
    test_docs = pd.read_csv('../data/test_docs.csv')
    title_seg = []

    # test_docs[test_docs['doc_title'].isnull()] = '无'
    # for target in test_docs['doc_title']:
    test_docs[test_docs['content'].isnull()] = '无'
    for target in test_docs['content']:
        seg_list = jieba.lcut(target, cut_all=False)
        title_seg.append(' '.join(seg_list))

    print(title_seg)
    # with open('./title_seg.o', 'wb') as p:
    with open('./content_seg.o', 'wb') as p:
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
    # test_data()
    # with open('./title_seg.o', 'rb') as r:
    #     title_seg = pickle.load(r)
    # print(title_seg)

    with open('./content_seg.o', 'rb') as r:
        content_seg = pickle.load(r)
    print(content_seg)

    # test_querys()

    with open('./querys_seg.o', 'rb') as r:
        title_seg = pickle.load(r)
    print(title_seg)

    corpus = content_seg + title_seg
    print(len(corpus))
    with open('./corpus.o', 'wb') as p:
        pickle.dump(corpus, p)
