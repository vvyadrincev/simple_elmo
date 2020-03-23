#!/bin/bash


input_dir=$1 # ../data/elmo_vectors_lemm_lower_true
model_dir=$2 # ../data/rusvectores_models/199

for fname in sentences_N.txt sentences_V.txt public_nouns.txt private_nouns.txt public_verbs.txt private_verbs.txt ; 
do 
    ./make_elmo_vectors_ruwordnet.py -i $input_dir/$fname -e $model_dir ; 
done
