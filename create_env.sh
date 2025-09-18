#!/bin/bash
set -euo pipefail

# Create conda env from environment.yml
conda env create -f environment.yml

# Extract environment name
ENV_NAME=$(head -n 1 environment.yml | cut -d' ' -f2)

echo "Environment '$ENV_NAME' created."
echo "Activate with: conda activate $ENV_NAME"