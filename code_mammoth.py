#%%
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


# %%
with open('.\\data\\mammoth_3d.json') as f:
  data = json.load(f)

# %%
df = pd.DataFrame (data, columns = ['x', 'y', 'z'])
df.head()
len(df)

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(90,72))

ax = fig.add_subplot(111, projection='3d')

ax.set_axis_off()
ax.scatter(df['x'], df['y'], df['z'], s=20, c='black')
ax.view_init(0, -170)

plt.show()

# %%
from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(n_clusters=12).fit(mammoth)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=45000, random_state=42) 
AC = AgglomerativeClustering(n_clusters=11, linkage='ward').fit(X_train)
labels = AC.labels_

KN = KNeighborsClassifier(n_neighbors=10).fit(X_train, labels)
labels_pred = KN.predict(df)
col_len = len(set(labels_pred))-1

# %%
from IPython.display import display
from IPython.display import Latex
pd.set_option('display.max_columns', 500)
from tqdm import tnrange, tqdm_notebook
%load_ext rpy2.ipython
# %%
%%R -i col_len -o color_scale

mycolors <- c('#FF0000', '#FF8000', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#FF00FF', '#00994C', '#000000', '#C0C0C0')
pal <- colorRampPalette(sample(mycolors))
color_scale <- sample(pal(col_len))
color_scale <- c(color_scale)

# %%
cmap = mpl.colors.ListedColormap(list(color_scale))
sns.palplot (color_scale)
# %%
from mpl_toolkits.mplot3d import Axes3D

# %%
fig = plt.figure(figsize=(40,30))

ax = fig.add_subplot(111, projection='3d')

ax.set_axis_off()
ax.scatter(df['x'], df['y'], df['z'], s=5, c=labels_pred, cmap=cmap)
ax.view_init(10, 190)

plt.show()
# %%
import umap.umap_ as umap

# %%
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1)
reducer.fit(df)

# %%
plt.figure(figsize=(40,30), facecolor='w')

plt.axis('off')
plt.scatter(reducer.embedding_[:, 0], reducer.embedding_[:, 1], s=5, c=labels_pred, cmap=cmap)
# %%
from scipy.spatial.distance import pdist
from sklearn.manifold._t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE

# %%
tsne = TSNE(n_components=2, perplexity=300).fit(df)

# %%
plt.figure(figsize=(25,25), facecolor='w')

plt.axis('off')
plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], s=5, c=labels_pred, cmap=cmap)

