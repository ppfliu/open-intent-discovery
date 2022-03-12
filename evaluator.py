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
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class Evaluator:
    def __init__(self, cluster_filepath, intent_filepath):
        self.cluster_filepath = cluster_filepath
        with open(intent_filepath) as intent_file:
            self.intent_dict = json.loads(intent_file.read())

    def evaluate(self):
        df = pd.read_csv(self.cluster_filepath, index_col=0)
        cluster_id_list = df.cluster_id.tolist()
        intent_id_list = []
        for intent in df.intent:
            intent_id_list.append(self.intent_dict[intent])

        target_names = self.intent_dict.keys()
        report = classification_report(intent_id_list, cluster_id_list, target_names=target_names, digits=3)
        print(report)

        confu_matrix = confusion_matrix(intent_id_list, cluster_id_list)
        print(confu_matrix)


if __name__=="__main__":
    cluster_filepath = 'results/snips_clusters.csv'
    intent_filepath = 'results/snips_intents.json'

    evaluator=Evaluator(cluster_filepath, intent_filepath)
    evaluator.evaluate()
