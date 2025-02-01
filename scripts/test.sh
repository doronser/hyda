#!/bin/bash

# Check if DOMAIN and CKPT_PATH arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 DOMAIN CKPT_PATH"
  exit 1
fi

# Assign the DOMAIN argument to a variable
DOMAIN=$1
CKPT_PATH=$2

# Run the command with the DOMAIN argument replacing SLIM
python train.py test -c brain_mri/configs/hyper.yaml \
  --ckpt_path ${CKPT_PATH} \
  --model.target_domain ${DOMAIN} \
  --data.target_domain ${DOMAIN} \
  hyper_${DOMAIN}_test