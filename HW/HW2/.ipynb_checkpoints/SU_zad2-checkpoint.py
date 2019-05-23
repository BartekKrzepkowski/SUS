import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kneed import KneeLocator
from functools import reduce
import webbrowser
import argparse
import glob
import cv2
import os

from umap import UMAP
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


def get_features_keras(paths):
    files = []
    model = VGG16(input_shape=(32,32, 3), include_top=False)
    for path in paths:
        img = image.load_img(path, target_size=(32, 32))
        file = np.expand_dims(image.img_to_array(img), axis=0)
        file = model.predict(file).squeeze()
        files.append(file)
    return np.array(files)


def get_compressed_features(features, metric, D):
    stsc = StandardScaler()
    files_sc = stsc.fit_transform(features)
    files_umap = UMAP(n_components=D, metric=metric, target_metric=metric).fit_transform(files_sc)
    stsc = StandardScaler()
    return stsc.fit_transform(files_umap)


def min_dist(indices, dist):
    N = dist.shape[0]
    k = (dist[N-1]-dist[0])/N
    f = lambda x: abs(dist[x] - k*x - dist[0])
    g = lambda x: k*x + dist[0]
    plt.figure(figsize=(10, 10))
    plt.scatter(indices, f(indices), label="dist from pend")
    plt.scatter(indices, g(indices), label="pend")
    plt.scatter(indices, dist, label="eps")
    pen = np.array(list(map(f, indices)))
    x_knee = np.argmax(pen)
    y_knee = round(dist[x_knee], 6)
    plt.axvline(x=x_knee, label="{} --x".format(x_knee))
    plt.axhline(y=y_knee, label="{} --y".format(y_knee))
    plt.legend()
    return x_knee, y_knee


def find_knee(data, k, metric, frac=1, kick=0):
    OUT = round(kick*data.shape[0]/1000)
    END = round(data.shape[0]*frac)
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(data)
    distances, _ = nbrs.kneighbors(data)
    distanceDec = np.array(sorted(distances[:, k-1], reverse=True))
    return min_dist(np.arange(END-OUT), distanceDec[OUT:END])


def display_clusters(grouped_idx, paths2files, browser="firefox", filename='HW2_332269_clusters.html'):
    g = open(filename, 'w')
    g.write('<html><head><h1>HW2 332269</h1></head><body>')
    for group in grouped_idx:
        g.write('<hr>')
        g.write('<p>')
        for img_idx in group:
            g.write('<img src={} style="display:inline" width="15" height="15" hspace="3" vspace="3">'.format(paths2files[img_idx]))
        g.write('</p>')
    g.write('</body></html>')
    g.close()
    webbrowser.get(browser).open_new_tab(filename)
        

def create_textfile(grouped_idx, paths2files, path):
    files_names = np.array(list(map(lambda x: x.replace(path+"/",""), paths2files)))
    with open("HW2_332269_clusters.txt", "w") as f:
        for group in grouped_idx:
            line = reduce(lambda x,y: x+" "+y, files_names[group])
            line += "\n"
            f.write(line)

    
def get_devided_idxs(paths2files, metric, minK):
    features = get_features_keras(paths2files)
    compressed_features = get_compressed_features(features, metric, D=minK-1)
    _, eps = find_knee(compressed_features, minK, metric, frac=0.9)
    db_clust = DBSCAN(eps=eps, min_samples=minK, n_jobs=-1, metric=metric)
    db_clust.fit(compressed_features)
    df = pd.DataFrame(list(enumerate(db_clust.labels_)), columns=["idx", "labels"]).groupby("labels").count().T
    return [el[1].index.values for el in pd.DataFrame(db_clust.labels_, columns=["labels"]).groupby("labels")]
    

def main(path, metric, minK):
    paths2files = np.array(glob.glob(os.path.join(path, "*.png")))
    grouped_idx = get_devided_idxs(paths2files, metric, minK)
    create_textfile(grouped_idx, paths2files, path)
    display_clusters(grouped_idx, paths2files)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path to the folder with images")
    parser.add_argument("-n", "--minK", required=True, type=int, help="minimal points in cluster")
    parser.add_argument("-m", "--metric", required=True, help="metric to compute distances")
    args = vars(parser.parse_args())

    main(**args)