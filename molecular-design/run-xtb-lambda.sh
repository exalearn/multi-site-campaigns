#! /bin/bash
search_space=../data/MOS-search.csv
# Get the MPNN models
mpnn_dir=../data/xtb-models
mpnn_files=$(find $mpnn_dir -name best_model.h5 | sort | tail -n 8)

# Relevant endpoints
#  2141035b-aeec-4163-9b5c-c23e4061710c: Debug queue, one node, flat memory
#  acdb2f41-fd86-4bc7-a1e5-e19c12d3350d: Lambda
#  0c1b0c51-2ae5-4401-8b0c-13cdb66c8e47: ThetaGPU single node
#  6c5c793b-2b2a-4075-a48e-d1bd9c5367f6: ThetaGPU full node

python run.py \
       --ml-endpoint acdb2f41-fd86-4bc7-a1e5-e19c12d3350d \
       --qc-endpoint 2141035b-aeec-4163-9b5c-c23e4061710c \
       --training-set $mpnn_dir/records.json \
       --retrain-from-scratch \
       --mpnn-model-files $mpnn_files \
       --search-space $search_space \
       --infer-ps-backend globus \
       --train-ps-backend globus \
       --simulate-ps-backend file \
       --ps-file-dir proxy-store-scratch \
       --ps-globus-config globus_config.json \
       $@
