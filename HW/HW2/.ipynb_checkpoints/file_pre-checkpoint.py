import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kneed import KneeLocator
from functools import reduce
import webbrowser
import argparse
import glob
import cv2

from umap import UMAP
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to the folder with images")
ap.add_argument("-n", "--minK", required=True, type=int, help="minimal points in cluster")
ap.add_argument("-m", "--metric", required=True, help="metric to compute distances")
args = vars(ap.parse_args())


def get_representation_keras(paths):
    files = []
    model = VGG16(input_shape=(32,32, 3), include_top=False)
    for path in paths:
        img = image.load_img(path, target_size=(32, 32))
        file = np.expand_dims(image.img_to_array(img), axis=0)
        file = model.predict(file).squeeze()
        files.append(file)
    return np.array(files)


def find_eps(data, k, metric):
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(data)
    distances, indices = nbrs.kneighbors(data)
    distanceDec = sorted(distances[:,k-1], reverse=True)
    knee = KneeLocator(indices[20:500,0], distanceDec[20:500], direction="decreasing", curve="convex")
    knee.plot_knee_normalized()
    return distanceDec[knee.elbow]


def create_file_display_clusters(grouped_idx, paths_files):
    g = open('HW2_BK_clusters.html','w')
    g.write('<html><head><h1>HW2 Bartlomiej Krzepkowski</h1></head><body>')
    pf = np.array(list(map(lambda x: x.replace(args["path"]+"/",""), paths_files)))
    with open("HW2_BK_clusters.txt", "w") as f:
        for group in grouped_idx[13:15]:
            line = reduce(lambda x,y: x+" "+y, pf[group])
            line += "\n"
            f.write(line)
            
            g.write("<hr>")
            g.write('<p>')
            for img_idx in group:
                g.write('<img src={} , " style="display: inline;", width="15", height="15">'.format(paths_files[img_idx]))
            g.write('</p>')
            
        g.write('</body></html>')
    
    g.close()
    webbrowser.open_new_tab('HW2_BK_clusters.html')
    

def main():
    metric = args["metric"]
    minK = args["minK"]
    D = minK - 1
    paths_files = glob.glob(args["path"] + "/*.png")
    paths_files = np.array(paths_files)
    files = get_representation_keras(paths_files)
    stsc = StandardScaler()
    files_sc = stsc.fit_transform(files)
    files_umap = UMAP(n_components=D, metric=metric, target_metric=metric).fit_transform(files_sc)
    stsc = StandardScaler()
    files_umap = stsc.fit_transform(files_umap)
    eps = find_eps(files_umap, minK, metric)
    db_clust = DBSCAN(eps=eps, min_samples=minK, n_jobs=-1, metric=metric)
    db_clust.fit(files_umap)
    df = pd.DataFrame(list(enumerate(db_clust.labels_)), columns=["idx", "labels"]).groupby("labels").count().T
    grouped_idx = [el[1].index.values for el in pd.DataFrame(db_clust.labels_, columns=["labels"]).groupby("labels")]
    
    create_file_display_clusters(grouped_idx, paths_files)
    
    
main()