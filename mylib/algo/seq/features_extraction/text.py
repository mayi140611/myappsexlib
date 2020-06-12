# AUTOGENERATED! DO NOT EDIT! File to edit: algo_seq_features_extraction_text.ipynb (unless otherwise specified).

__all__ = ['buildTfidf']

# Cell
def buildTfidf(texts):
    """
    构建tfidf矩阵
    :param texts:
        [
            'w1 w2 w3',
            'w4 w5',
            ...
        ]
    :return:
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    print('building tfidf array')
    tfidf = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b[\w\.\+\-/]+\b")  # 匹配字母数字下划线和小数点  如10.0
    tfidf_features = tfidf.fit_transform(texts)
    print('building tfidf array completed')
    return tfidf_features, tfidf