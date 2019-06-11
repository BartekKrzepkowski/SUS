import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

def unnest(df, col):
    unnested = (df.apply(lambda x: pd.Series(x[col]), axis=1)
                .stack()
                .reset_index(level=1, drop=True))
    unnested.name = col
    return df.drop(col, axis=1).join(unnested)


def to_bag_of_cards(df):
    df['ind'] = np.arange(df.shape[0]) + 1
    df_orig = df.copy()
    df['deck'] = df['deck'].apply(lambda d: d.split(';'))
    df = unnest(df, 'deck')
    df['value'] = 1
    df_bag = df.pivot(index='ind', columns='deck', values='value')
    df_bag[df_bag.isna()] = 0
    df_bag = df_bag.astype('int')
    return pd.concat([df_orig.set_index('ind'), df_bag], axis=1)


def get_n_from_cluster(data, labels, feat):
    top_1500 = []
    for i in range(1000):
        df_cluster = data[labels == i]
        chosen = df_cluster.sort_values(by=feat, ascending=False)[: 10]
        top_1500.append(chosen)
    top_1500 = pd.concat(top_1500)
    top_1500 = top_1500.sort_values(by=feat, ascending=False)[: 1500]
    return top_1500.index.values - 1


def get_data(train, valid, feat):
    train = to_bag_of_cards(train).drop(["deck"], axis=1)
    valid = to_bag_of_cards(valid).drop(["deck"], axis=1)
    pca = PCA(n_components=10)
    x_train_10 = pca.fit_transform(train.values)
    
    knn = MiniBatchKMeans(n_clusters=1000)
    knn.fit(x_train_10)
    
    return train, valid, get_n_from_cluster(train, knn.labels_, feat)


def R2_1(y, x):
    return 1 - np.sum(np.square(x - y)) / np.sum(np.square(y - np.mean(y)))


def R2_2(y, x):
    return np.sum(np.square(x - np.mean(y))) / np.sum(np.square(y - np.mean(y)))


def query_strategy(regressor, X):
    std = regressor.predict(X)
    out = np.concatenate([np.arange(std.shape[0])[:, np.newaxis], std[:, np.newaxis]], axis=1)
    out = out[out[:, 1].argsort()]
    query_idx = out[:100, 0].astype(int)
    return query_idx, X[query_idx]


def round_params(params):
    for label in params:
        params[label] = np.round(params[label], 4)
    return params