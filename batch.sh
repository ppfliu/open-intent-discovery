#!/bin/bash

results_folder=results
mkdir -p $results_folder

for dataset_file in data/snips.csv
do
    echo $dataset_file
    for nclusters in {5..10}
    do
        for coef in 0.1
        do
            for method in kmeans
            do
                for bert_type in paraphrase-distilroberta-base-v1 nli-bert-base-max-pooling stsb-roberta-base
                do
                    echo "Running $method using $bert_type on $dataset_file: ##################################"
                    CUDA_VISIBLE_DEVICES=-1 python main.py --vector_type=BERT --bert_type=$bert_type --coef_lambda=$coef \
                        --method=$method --nclusters=$nclusters --dataset_file=$dataset_file --results_folder=$results_folder >> $method-$bert_type-$coef.log
                done
            done
        done
    done
done

