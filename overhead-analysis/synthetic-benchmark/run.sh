#!/bin/bash
#COBALT -A CSC249ADCD08
#COBALT -t 60
#COBALT -n 1
#COBALT -q debug-flat-quad
#COBALT --attrs enable_shh=0

CONFIG='config.json'

module load miniconda-3/latest
conda activate colmena

# Start the redis server
PORT=59465
redis-server --port $PORT --protected-mode no &> redis.out &
REDIS=$!

echo "Redis started on $HOSTNAME:$PORT"

# Personal Theta Login Node Endpoint
ENDPOINT=5c39db12-693b-4803-ad5b-1582b6111ab4
# ThetaGPU Full Node Endpoint
# ENDPOINT=6d56207b-67f6-4f08-9805-af337e0bea6c

INPUT_SIZE_MB=100
OUTPUT_SIZE_MB=0
DURATION=0
TASKS=20
INTERVAL=5
PS_BACKEND=globus

python synthetic.py \
	--redis-host $HOSTNAME \
	--redis-port $PORT \
    --endpoint $ENDPOINT \
	--task-input-size $INPUT_SIZE_MB \
	--task-output-size $OUTPUT_SIZE_MB \
	--task-interval $INTERVAL \
	--task-count $TASKS \
    --task-time $DURATION \
    --ps-backend $PS_BACKEND \
    --ps-threshold 0.005 \
    --ps-file-dir /lus/theta-fs0/projects/CSC249ADCD08/jgpaul/scratch/proxystore-dump/ \
    --ps-globus-config globus_config.json \
    --output-dir "runs/${PS_BACKEND}/${TASKS}T-${DURATION}D-${INPUT_SIZE_MB}I-${OUTPUT_SIZE_MB}O-1N" \

# Kill the redis server
kill $REDIS

