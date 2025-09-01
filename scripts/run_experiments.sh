#!/bin/bash

MODELS=(
  "opensearch-project/opensearch-neural-sparse-encoding-doc-v1"
  "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
  "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
  "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
)

DATASETS=("limit-small" "limit")

for dataset in "${DATASETS[@]}"
do
  for model in "${MODELS[@]}"
  do
    echo "Running experiment with model: $model on dataset: $dataset"
    model_name=$(echo "$model" | cut -d'/' -f2)
    python sparse_embedding.py --model "$model" --dataset "$dataset" > "logs/${model_name}_${dataset}.log"
    echo "Finished experiment with model: $model on dataset: $dataset"
    echo "--------------------------------------------------"
  done
done

echo "All experiments finished."
