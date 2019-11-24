import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import pandas as pd


docs_len = 13213
querys_len = 470


def tfidf():
    with open('./corpus.o', 'rb') as r:
        corpus = pickle.load(r)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    print(len(word))
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    print(tfidf)
    print(tfidf.shape)
    with open('./tfidf.o', 'wb') as p:
        pickle.dump(tfidf, p)


def idx():
    with open('./tfidf.o', 'rb') as r:
        tfidf = pickle.load(r)
    tfidf_docs = tfidf[:13213, :]
    tfidf_querys = tfidf[13213:, :]
    tfidf_docs_norm = normalize(tfidf_docs)
    tfidf_querys_norm = normalize(tfidf_querys)
    cos = tfidf_querys_norm * tfidf_docs_norm.T
    # print(cos)

    cos = cos.toarray()
    print(cos.shape)
    idx = np.argsort(-cos, axis=1)[:, :20]
    print(idx)
    print(idx.shape)

    with open('./idx.o', 'wb') as p:
        pickle.dump(idx, p)


if __name__ == '__main__':
    with open('./idx.o', 'rb') as r:
        idx = pickle.load(r)
    print(idx)
    print(idx.shape)
    test_docs = pd.read_csv('../data/test_docs.csv')
    test_querys = pd.read_csv('../data/test_querys.csv')

    for i in range(idx.shape[0]):
        print(test_querys['query'][i])
        for j in range(idx.shape[1]):
            print('\t' + test_docs['doc_title'][idx[i][j]])
        print()
