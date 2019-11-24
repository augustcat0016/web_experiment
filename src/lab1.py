import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
#read csv
querys = pd.read_csv("test_querys.csv")
docs = pd.read_csv("test_docs.csv")
querys = querys.astype(str)
docs = docs.astype(str)

#分词
def cutword(x):
    seg = jieba.cut(x)
    return ' '.join(seg)


doc_query = querys['query'].apply(cutword)
documents = docs['content'].apply(cutword)
all_doc = documents.append(doc_query, ignore_index=True)
#计算tfidf矩阵
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(all_doc))
weight = tfidf.toarray()
weight_query = weight[:querys.shape[0]]
weight_doc = weight[querys.shape[0]:]
#矩阵正则化，用于计算相似度作准备
weight_query = normalize(weight_query, norm='l2')
weight_doc = normalize(weight_doc, norm='l2')

#保存矩阵
np.save('weight_query.npy', weight_query)
np.save('weight_doc.npy', weight_doc)