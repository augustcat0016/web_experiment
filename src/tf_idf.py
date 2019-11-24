import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == '__main__':
    with open('./corpus.o', 'rb') as r:
        corpus = pickle.load(r)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    print(len(word))
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    tfidf = np.array(tfidf)
    print(tfidf)
    print(tfidf.shape)
    with open('./tfidf.o', 'wb') as p:
        pickle.dump(tfidf, p)
