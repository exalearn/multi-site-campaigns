#! /bin/bash
search_space=../data/MOS-search.csv

# Get the MPNN models
mpnn_dir=../data/xtb-models
mpnn_files=$(find $mpnn_dir -name best_model.h5 | sort | tail -n 8)

# Relevant endpoints
#  b9c5db67-cc17-4708-be97-5274443d789a: Debug queue, one node, flat memory
#  acdb2f41-fd86-4bc7-a1e5-e19c12d3350d: Lambda
#  de92a1ac-4118-48a2-90ac-f43a59298634: Venti

python run.py \
       --ml-endpoint de92a1ac-4118-48a2-90ac-f43a59298634 \
       --qc-endpoint b9c5db67-cc17-4708-be97-5274443d789a \
       --training-set $mpnn_dir/records.json \
       --retrain-from-scratch \
       --mpnn-model-files $mpnn_files \
       --search-space $search_space \
       --infer-ps-backend redis \
       --train-ps-backend redis \
       --simulate-ps-backend file \
       --ps-file-dir proxy-store-scratch \
       --ps-globus-config globus_config.json \
       --num-qc-workers 8 \
       --retrain-frequency 1 \
       --molecules-per-ml-task 100000 \
       --search-size 256 \
       --use-parsl
