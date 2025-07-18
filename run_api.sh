#!/bin/bash
# Run API with GPU
export USE_GPU=1
python api.py

# Or without GPU
# export USE_GPU=0
# python api.py