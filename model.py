# Copyright (C) 2021 Pengfei Liu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

from numba import jit

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

@jit
def kmeans(vectors, nclusters):
    cluster_model = KMeans(n_clusters=nclusters, init='k-means++', random_state=42).fit(vectors)
    return cluster_model, cluster_model.labels_ #, cluster_model.cluster_centers_

def balance_penalty(cluster_id, nclusters, coef_lambda):
    N = len(cluster_id)
    t = nclusters

    df=pd.DataFrame()
    df['num'] = cluster_id.tolist()
    counts = pd.DataFrame(df['num'].value_counts())['num'].tolist()

    # calculate balance
    balance=0
    for item in counts:
        balance = balance + abs(item/float(N)-1/float(t))

    # calculate mean and sigma
    square_sum = 0
    miu = N/t
    for item in counts:
        square_sum += np.square(item-miu)

    sigma=np.sqrt(square_sum/float(t-1))
    print("Cluster sizes: mean={:.5f}, sigma={:.5f}, sigma/mean={:.5f}".format(N/t, sigma, sigma/miu))

    penalty = coef_lambda * (sigma/miu) * balance
    return penalty

# Cluster Model
class ClusterModel:
    def __init__(self, vector_type='TFIDF', bert_type=None, method='kmeans'):
        self.vector_type = vector_type
        self.bert_type = bert_type
        self.vectors = None

        self.method = method
        self.cluster_model = None
        self.cluster_id = None

    def vectorize(self, sentences):
        if self.vector_type == 'TFIDF':
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(sentences)
            print('TF-IDF vectors of shape:', vec.shape)
            return vec.A
        elif self.vector_type == 'BERT':
            if self.bert_type == "USE":
                import tensorflow_hub as hub
                use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                vec = np.array(use_model(sentences))
            else:
                sent_model = SentenceTransformer(self.bert_type)
                vec = np.array(sent_model.encode(sentences, show_progress_bar=True))

            print('BERT vectors of shape:', vec.shape)
            return vec

    def fit(self, sentences, coef_lambda=0.1, nclusters=7):
        # get vecors
        if not self.vectors:
            self.vectors = self.vectorize(sentences)

        # run clustering
        if self.method=='kmeans':
            self.cluster_model, self.cluster_id = kmeans(self.vectors, nclusters)
        elif self.method=='hier':
            dist = pdist(self.vectors, metric='sqeuclidean')
            linkage_matrix = ward(dist)
            self.cluster_model = linkage_matrix
            self.cluster_id = fcluster(self.cluster_model, nclusters, criterion='maxclust') - 1
        elif self.method=="gmm":
            gmm = GaussianMixture(n_components=nclusters, random_state=42)
            self.cluster_model = gmm.fit(self.vectors)
            self.cluster_id = self.cluster_model.predict(self.vectors)
        else:
            print('Not support the clustering method: {}'.format(self.method))

        sil_score = silhouette_score(self.vectors, self.cluster_id, metric='euclidean')

        penalty = balance_penalty(self.cluster_id, nclusters, coef_lambda)
        balance_score = sil_score - penalty
        print('Method: {}, Silhouette: {:.5f}, Balance: {:.5f}, Penalty: {:.5f}'.format(
                self.method, sil_score, balance_score, penalty))
