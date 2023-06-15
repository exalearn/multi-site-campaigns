#! /bin/bash
search_space=/lus/theta-fs0/projects/CSC249ADCD08/multi-site-campaigns/data/moldesign/search-space/MOS-search.csv

# Get the MPNN models
data_dir=/lus/theta-fs0/projects/CSC249ADCD08/multi-site-campaigns/data/moldesign
mpnn_file=$data_dir/initial-model/networks/gpu_b16_n256_Rsum_cbd46e/model.h5

echo $mpnn_file mpnn


# Relevant endpoints
#  50598bb9-020a-4a5c-a62a-c79304521ee3: Debug queue, one node, flat memory
#  cc4860b3-b170-478b-9bff-603aa75b65c8: Chameleon Cloud

python run.py \
       --ml-endpoint e670dbe9-bd11-4380-b7de-9ac2dda133df \
       --qc-endpoint dea3da25-763f-436a-b667-bae3fa7f0f52  \
       --redisport 6379 \
       --training-set $data_dir/training-data.json \
       --mpnn-model-path $mpnn_file \
       --model-count 8 \
       --num-epochs 128 \
       --search-space $search_space \
       --infer-ps-backend multi \
       --train-ps-backend multi \
       --simulate-ps-backend multi \
       --ps-file-dir proxy-store-scratch \
       --ps-multi-config multistore_zmq_endpoint.json \
       --num-qc-workers 8 \
       --retrain-frequency 1 \
       --molecules-per-ml-task 50000 \
       --search-size 512 \
       --ps-threshold 10000
