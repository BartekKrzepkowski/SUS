from kneed import KneeLocator


def find_eps(data, k, metric):
    END = round(data.shape[0]*0.9)
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(data)
    distances, indices = nbrs.kneighbors(data)
    distanceDec = np.array(sorted(distances[:,k-1], reverse=True))
    knee = KneeLocator(indices[:END, 0], distanceDec[:END], curve="convex", direction="decreasing", S=1.0)
    knee.plot_knee_normalized()
    return distanceDec[knee.elbow], knee



def display_cluster(paths_files, cluster_idx, nb=200):
    permuted_idx = np.random.permutation(cluster_idx)
    pf = paths_files[permuted_idx]
    display_images(pf, nb=nb, w=20)
    
    
# SPRÓBUJ HDBSCAN
from hdbscan import HDBSCAN

# HDBSCAN
hdb_clust = HDBSCAN(min_cluster_size=20, min_samples=minK, core_dist_n_jobs=-1, metric="minkowski", allow_single_cluster=True, p=2)
hdb_clust.fit(files_umap)
df = pd.DataFrame(list(enumerate(hdb_clust.labels_)), columns=["idx", "labels"]).groupby("labels").count().T
grouped_idx = [el[1].index.values for el in pd.DataFrame(hdb_clust.labels_, columns=["labels"]).groupby("labels")]
display(HTML(df.to_html()))

from sklearn import OPTICS

# OPTICS
opt_clust = OPTICS(min_samples=minK, n_jobs=-1, metric=metric_c)
opt_clust.fit(files_umap)
df = pd.DataFrame(list(enumerate(db_clust.labels_)), columns=["idx", "labels"]).groupby("labels").count().T
grouped_idx = np.array([el[1].index.values for el in pd.DataFrame(db_clust.labels_, columns=["labels"]).groupby("labels")])
display(HTML(df.to_html()))