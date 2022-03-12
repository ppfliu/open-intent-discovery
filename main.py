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

import os
import argparse

import warnings
warnings.filterwarnings('ignore', category=Warning)

import pandas as pd

from model import ClusterModel
from extractor import Extractor
from utils import preprocess, draw_tsne
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str,default='./data/snips.csv')
    parser.add_argument('--nclusters', type=int, default=None)
    parser.add_argument('--coef_lambda', type=float, default=0.1)
    parser.add_argument('--vector_type', type=str, default='BERT')
    parser.add_argument('--method', type=str, default='kmeans', help='kmeans, hier, gmm')
    parser.add_argument("--bert_type", type=str, default='paraphrase-distilroberta-base-v1')
    parser.add_argument("--results_folder", type=str, default='./results/')

    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    # Print dataset information
    dataset_name = os.path.basename(args.dataset_file)[:-4]
    data = pd.read_csv(str(args.dataset_file))
    data = data.fillna('')

    intent_set = sorted(list(set(data.intent)))
    intent_numlist = [intent_set.index(item) for item in data.intent]
    sentences, idx_in, intent_list = preprocess(data.text, data.intent)

    print('Dataset: {}, #text: {}, #sentences: {}'.format(dataset_name, len(data.text), len(sentences)))
    print('#intents: {}, intent-set: {}'.format(len(intent_set), ' '.join(intent_set)))

    # Fit the cluster model
    model = ClusterModel(vector_type=str(args.vector_type), bert_type=str(args.bert_type), method=str(args.method))

    nclusters = args.nclusters if args.nclusters else len(intent_set)
    model.fit(sentences, args.coef_lambda, nclusters)

    # Print clustering performance
    nmi = normalized_mutual_info_score(intent_numlist, model.cluster_id)
    ari = adjusted_rand_score(intent_numlist, model.cluster_id)
    print("Dataset: {}, #intents:{}, nmi:{:.5f}, ari:{:.5f}".format(dataset_name, len(intent_set), nmi, ari))

    # Save clustering results
    result = pd.DataFrame()
    result['intent'] = intent_list
    result['text'] = sentences
    result['cluster_id'] = model.cluster_id
    result.index = idx_in
    result.index.name = 'id'
    result.sort_values(['cluster_id'], inplace=True, ascending=True)

    cluster_filepath = '{}/{}_clusters.csv'.format(args.results_folder, dataset_name)
    result.to_csv(cluster_filepath)

    # Plot t-SNE for clusters
    fig_filepth = '{}/{}_tsne.pdf'.format(args.results_folder, dataset_name)
    draw_tsne(model.vectors, model.cluster_id, fig_filepth)

    # Extract action-object pairs for each cluster
    parser_filepath = '{}/{}_parser.csv'.format(args.results_folder, dataset_name)
    intent_filepath = '{}/{}_intents.json'.format(args.results_folder, dataset_name)
    extractor = Extractor(in_cluster_filepath = cluster_filepath,
                          out_parser_filepath = parser_filepath,
                          out_intent_filepath = intent_filepath)
    extractor.get_action_object_pairs()