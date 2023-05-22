import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime


def get_reduced_data(source_data, method, components=2, scale=False, scale_fit=True):
    reduction_methods = DimensionReduction()
    if method == 'PCA':
        data = reduction_methods.PCA(source_data, scale=scale, scale_fit=scale_fit)[0:60000]
    elif method == 'UMAP':
        data = reduction_methods.UMAP(source_data, scale=scale, scale_fit=scale_fit)[0:60000]
    elif method == 'TSNE':
        data = reduction_methods.TSNE(source_data, scale=scale, scale_fit=scale_fit)[0:60000]

    if isinstance(data, pd.DataFrame):
        data = data.values
    if data.shape[1] <= components:
        x = data
    else:
        x = data[:, 0:components]

    if method is None:
        x = source_data
    return x


class DimensionReduction:
    pca_scaler = MinMaxScaler()
    umap_scaler = MinMaxScaler()
    tsne_scaler = MinMaxScaler()

    def __init__(self):
        pass

    def PCA(self, data, scale=False, scale_fit=True):
        pca = PCA(random_state=42)
        Xt = pca.fit_transform(data)
        if scale_fit:
            DimensionReduction.pca_scaler = MinMaxScaler().fit(Xt)
        if scale:
            Xt = pd.DataFrame(DimensionReduction.pca_scaler.transform(Xt))
        print("PCA explained variance ratio: ", pca.explained_variance_ratio_)
        return Xt

    def UMAP(self, data, scale=False, scale_fit=True):
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=42)
        # reducer = umap.UMAP()
        umap_data = reducer.fit_transform(data)
        if scale_fit:
            DimensionReduction.umap_scaler = MinMaxScaler().fit(umap_data)
        if scale:
            umap_data = pd.DataFrame(DimensionReduction.umap_scaler.transform(umap_data))
        print("Umap model shape: ", umap_data.shape)
        return umap_data

    def TSNE(self, data, scale=False, scale_fit=True):
        reducer = TSNE(random_state=42)
        tsne_data = reducer.fit_transform(data)
        if scale_fit:
            DimensionReduction.umap_scaler = MinMaxScaler().fit(tsne_data)
        if scale:
            tsne_data = pd.DataFrame(DimensionReduction.umap_scaler.transform(tsne_data))
        print("TSNE model shape: ", tsne_data.shape)
        return tsne_data

    def dimension_reduction_visualization(self, data, method):
        if method == 'UMAP':
            x = self.UMAP(data)
        elif method == 'PCA':
            x = self.PCA(data)
        elif method == 'TSNE':
            x = self.TSNE(data)

        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1])
        plt.xlabel("component_1")
        plt.ylabel("component_2")
        # plt.title("First two " + method + " components")
        plt.title(method)
        plt.savefig(datetime.now().strftime("%H:%M:%S") + '.eps')
        plt.show()
