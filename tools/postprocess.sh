#!/bin/bash


green=`tput setaf 2`
reset=`tput sgr0`

CORPUS_DIR=${1:-cora_enrich}

# python postprocess.py --dataset ${CORPUS_DIR} --in_file phrase_text.txt --out_file all_text.txt

echo ${green}===Embedding Learning on Full Text===${reset}
CORPUS_FILE=phrase_text.txt
OUTPUT=all
./fasttext skipgram -input ${CORPUS_DIR}/${CORPUS_FILE} -output ${CORPUS_DIR}/${OUTPUT} -dim 250

