from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers.sparse_encoder import SparseEncoder
import logging
import os
import torch
import pathlib
import time

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

import argparse

parser = argparse.ArgumentParser(
    description="Evaluate a sparse retrieval model on the LIMIT dataset."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="limit-small",
    help="Name of the dataset to use (e.g., limit-small or limit)",
)
parser.add_argument(
    "--model",
    type=str,
    default="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
    help="Name of the sparse model to use from Hugging Face.",
)
args = parser.parse_args()

dataset = args.dataset
data_path = pathlib.Path(__file__).parent / "data" / dataset

if not os.path.exists(data_path):
    dataset_folder = data_path.name
    # Download and unzip the dataset if it doesn't exist
    url = f"https://huggingface.co/datasets/orionweller/{dataset_folder.upper()}/resolve/main/{dataset_folder}.zip"
    util.download_and_unzip(url, str(data_path.parent))

import json


# Custom data loading functions
def load_qrels(path):
    qrels = {}
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = int(row["score"])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score
    return qrels


def load_corpus(path):
    corpus = {}
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
    return corpus


def load_queries(path):
    queries = {}
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]
    return queries


corpus = load_corpus(os.path.join(data_path, "corpus.jsonl"))
queries = load_queries(os.path.join(data_path, "queries.jsonl"))
qrels = load_qrels(os.path.join(data_path, "qrels.jsonl"))

# https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v1

if args.model == "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte":
    model = SparseEncoder(
        args.model,
        trust_remote_code=True,
        model_kwargs={"code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"},
    )
else:
    model = SparseEncoder(args.model, trust_remote_code=True)

results = {}
# Encode the corpus
corpus_ids = list(corpus.keys())
corpus_texts = [corpus[cid]["title"] + " " + corpus[cid]["text"] for cid in corpus_ids]
start_time = time.time()
corpus_embeds = model.encode_document(corpus_texts, show_progress_bar=True)
end_time = time.time()
print(f"Corpus encoding time: {end_time - start_time:.4f} seconds")

# Encode queries
query_ids = list(queries.keys())
query_texts = [queries[qid] for qid in query_ids]
start_time = time.time()
query_embeds = model.encode_query(query_texts, show_progress_bar=True)
end_time = time.time()
print(f"Query encoding time: {end_time - start_time:.4f} seconds")

from tqdm.autonotebook import trange

# Perform retrieval
# Transpose corpus embeddings for efficient matrix multiplication
corpus_embeds_t = corpus_embeds.T

for i in trange(len(query_embeds), desc="Scoring"):
    query_id = query_ids[i]
    query_embed = query_embeds[i].to_dense()

    # Compute scores using sparse matrix multiplication
    scores = torch.matmul(query_embed, corpus_embeds_t)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Rank documents
    ranked_doc_indices = scores.argsort()[::-1]

    # Store results
    results[query_id] = {
        corpus_ids[doc_idx]: float(scores[doc_idx])
        for doc_idx in ranked_doc_indices[:100]
    }

#### Evaluate your model with NDCG@10
k_values = [2, 10, 20, 100]
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
print("NDCG@10: {:.4f}".format(ndcg["NDCG@10"]))
for k in k_values:
    print(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")
