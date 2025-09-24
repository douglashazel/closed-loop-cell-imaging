#!/bin/bash
# create_env.sh

YML_FILE=environment.yml

# optional: remove any existing env with the name in the yml
ENV_NAME=$(grep -m1 '^name:' "$YML_FILE" | awk '{print $2}')
conda env remove -n "$ENV_NAME" -y 2>/dev/null

conda env create -f "$YML_FILE"

echo "Environment '$ENV_NAME' created."
echo "Activate it with: conda activate $ENV_NAME"
