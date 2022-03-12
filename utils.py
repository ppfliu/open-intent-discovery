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

import time
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.manifold import TSNE


def preprocess(docs, intents):
    sentences = []
    idx_in = []
    intent_list=[]

    for idx in range(len(docs)):
        idx_in.append(idx)
        sentences.append(docs[idx].lower())
        intent_list.append(intents[idx])

    print("Preprocessed #sentences:", len(sentences))
    return sentences, idx_in, intent_list

def draw_tsne(vec, labels, fig_path):
    time_start = time.time()
    tsne=TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000, n_jobs=6)
    embedding = tsne.fit_transform(vec)

    n = len(vec)
    counter = Counter(labels)
    for i in range(len(np.unique(labels))):
        plt.plot(embedding[:, 0][labels == i], embedding[:, 1][labels == i], '.', alpha=0.5,
                 label='{:.2f}%'.format(counter[i] / n * 100))
    plt.legend(loc='lower right', fontsize="medium", labelspacing=0.3, borderpad=0.2)
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Plot t-SNE figure at {}, in {:.3f} seconds'.format(fig_path, time.time()-time_start))
