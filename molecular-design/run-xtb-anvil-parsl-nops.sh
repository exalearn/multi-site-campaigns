#! /bin/bash
search_space=/lus/theta-fs0/projects/CSC249ADCD08/multi-site-campaigns/data/moldesign/search-space/MOS-search.csv

# Get the MPNN models
data_dir=/lus/theta-fs0/projects/CSC249ADCD08/multi-site-campaigns/data/moldesign
mpnn_file=$data_dir/initial-model/networks/gpu_b16_n256_Rsum_cbd46e/model.h5


# Relevant endpoints
#  dea3da25-763f-436a-b667-bae3fa7f0f52: Debug queue, one node, flat memory
#  acdb2f41-fd86-4bc7-a1e5-e19c12d3350d: Lambda

python run.py \
       --ml-endpoint ae738989-2bd6-45a5-a823-077295457064 \
       --qc-endpoint 50598bb9-020a-4a5c-a62a-c79304521ee3 \
       --redisport 6379 \
       --training-set $data_dir/training-data.json \
       --mpnn-model-path $mpnn_file \
       --model-count 8 \
       --num-epochs 128 \
       --search-space $search_space \
       --no-proxystore \
       --ps-file-dir proxy-store-scratch \
       --ps-multi-config /home/vhayot/multi-site-campaigns/configs/multistore_zmq_endpoint.json \
       --num-qc-workers 8 \
       --retrain-frequency 1 \
       --molecules-per-ml-task 50000 \
       --search-size 512 \
       --ps-threshold 10000 \
       --use-parsl
       #--infer-ps-backend multi \
       #--train-ps-backend multi \
       #--simulate-ps-backend multi \
