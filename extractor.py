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


import json
import spacy
import pandas as pd

from collections import Counter


class Extractor:
    def __init__(self, in_cluster_filepath, out_parser_filepath, out_intent_filepath):
        self.in_cluster_filepath = in_cluster_filepath
        self.out_parser_filepath = out_parser_filepath
        self.out_intent_filepath = out_intent_filepath

        self.nlp = spacy.load("en_core_web_sm")

    def get_action_object_pairs(self):
        df = pd.read_csv(self.in_cluster_filepath)

        action_words = []
        object_words = []
        action_object_pairs = []
        for item in df.text:
            deps = spacy.displacy.parse_deps(self.nlp(item))
            action, object = 'NONE', 'NONE' # default
            for arc in deps['arcs']:
                if arc['label'] == 'dobj':
                    start = deps['words'][arc['start']]
                    if start['tag'] == 'VERB':
                        action = start['text'].lower()

                    end = deps['words'][arc['end']]
                    if end['tag'] == 'NOUN':
                        object = end['text'].lower()

                    # print(action, object)
                    continue

            action_words.append(action)
            object_words.append(object)
            action_object_pairs.append('{}-{}'.format(action, object))

        df['action'] = action_words
        df['object'] = object_words
        df['pair'] = action_object_pairs
        df.to_csv(self.out_parser_filepath, index=False)

        cluster_id_set = set(df.cluster_id.tolist())
        for cluster_id in cluster_id_set:
            df_subset = df.loc[(df['cluster_id'] == cluster_id) & (~df['pair'].str.contains('NONE'))]
            # action_counter = Counter(df_subset['action'].tolist())
            # object_counter = Counter(df_subset['object'].tolist())
            pair_counter = Counter(df_subset['pair'].tolist())
            
            # print('Top actions for cluster {}: {}'.format(cluster_id, action_counter.most_common(5)))
            # print('Top objects for cluster {}: {}'.format(cluster_id, object_counter.most_common(5)))
            print('Top pairs for cluster {}: {}'.format(cluster_id, pair_counter.most_common(5)))

        intent_dict = {}
        for idx, intent in enumerate(sorted(set(df.intent.tolist()))):
            intent_dict[intent] = idx

        print(intent_dict)
        with open(self.out_intent_filepath, "w") as intent_file:
            json.dump(intent_dict, intent_file, sort_keys=True, indent=4)

if __name__=="__main__":
    # Extract action-object pairs
    extractor = Extractor(in_cluster_filepath = 'results/snips_clusters.csv',
                          out_parser_filepath = 'results/snips_parser.json',
                          out_intent_filepath = 'results/snips_intents.json')
    extractor.get_action_object_pairs()
