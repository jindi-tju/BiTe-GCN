#!/bin/bash
DATASET=hep-small
# python generate_binary.py --dataset ${DATASET}
python train.py --dataset_str ${DATASET} \
                --dataset data/word_data/${DATASET}/${DATASET}.pickle.bin \
                --hidden_dim 64 \
                --out_dim 3 \
                --model HeteroGAT \
                --node_feature raw \
                --log_interval 1 \
                --aggr_func mean \
                --num_layer 2 \
                --num_head 2 \
                --residual 1 \
                --patience 25 \
                --word_feature w2v \
                --device cuda