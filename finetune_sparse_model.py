import logging
from datasets import Dataset
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.losses import (
    SpladeLoss,
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers
import os
import pathlib
import json
import argparse
from beir.retrieval.evaluation import EvaluateRetrieval
import torch
from tqdm.autonotebook import trange
import matplotlib.pyplot as plt
import random
from sparse_embedding import load_corpus, load_queries, load_qrels

random.seed(42)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune a sparse retrieval model on the LIMIT dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="limit",
        help="Name of the TEST dataset to use (e.g., limit-small or limit)",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="limit-train",
        help="Name of the TRAIN dataset to use (e.g., limit-train)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
        help="Name of the sparse model to use from Hugging Face.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/limit-finetuned-model",
        help="Output directory to save the finetuned model and logs.",
    )
    args = parser.parse_args()

    # Create output directory, if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Training Data ---
    train_data_path = pathlib.Path(__file__).parent / "data" / args.train_dataset
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Training dataset not found at {train_data_path}. Please run build_train_set.py first."
        )

    train_corpus = load_corpus(os.path.join(train_data_path, "corpus.jsonl"))
    train_queries = load_queries(os.path.join(train_data_path, "queries.jsonl"))
    train_qrels = load_qrels(os.path.join(train_data_path, "qrels.jsonl"))

    all_training_samples = []
    for query_id, docs in train_qrels.items():
        query_text = train_queries[query_id]
        for doc_id, score in docs.items():
            if score > 0:
                doc = train_corpus[doc_id]
                doc_text = doc["title"] + " " + doc["text"]
                all_training_samples.append({"query": query_text, "document": doc_text})

    random.shuffle(all_training_samples)

    # --- Load Test Data ---
    test_data_path = pathlib.Path(__file__).parent / "data" / args.dataset
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test dataset not found at {test_data_path}.")

    test_corpus = load_corpus(os.path.join(test_data_path, "corpus.jsonl"))
    test_queries = load_queries(os.path.join(test_data_path, "queries.jsonl"))
    test_qrels = load_qrels(os.path.join(test_data_path, "qrels.jsonl"))

    # --- Experiment Loop ---
    training_amounts = [2, len(all_training_samples)]
    recall_at_10_scores = []
    final_model = None

    for num_samples in training_amounts:
        logging.info(f"--- Training with {num_samples} samples ---")

        # Create output directory for current sample training
        sample_output_dir = os.path.join(args.output_dir, f"sample_{num_samples}")
        os.makedirs(sample_output_dir, exist_ok=True)

        training_subset = all_training_samples[:num_samples]
        train_dataset = Dataset.from_list(training_subset)
        logging.info(f"Training subset created with {len(train_dataset)} samples.")

        if (
            args.model
            == "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
        ):
            model = SparseEncoder(
                args.model,
                trust_remote_code=True,
                model_kwargs={
                    "code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"
                },
            )
        else:
            model = SparseEncoder(args.model, trust_remote_code=True)

        loss = SpladeLoss(
            model=model,
            loss=SparseMultipleNegativesRankingLoss(model=model),
            query_regularizer_weight=5e-5,
            document_regularizer_weight=3e-5,
        )

        # 40 steps
        if num_samples == 2:
            num_train_epochs = 40
            per_device_train_batch_size = 1
        elif num_samples == len(all_training_samples):
            num_train_epochs = 4
            per_device_train_batch_size = 85

        training_args = SparseEncoderTrainingArguments(
            output_dir=sample_output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            warmup_ratio=0.1,
            learning_rate=5e-5,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=100,
            run_name=f"{args.model.replace('/', '_')}-{args.dataset}-samples-{num_samples}",
            router_mapping={"query": "query", "document": "document"},
        )

        trainer = SparseEncoderTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
        )
        trainer.train()

        logging.info(f"Starting evaluation on the {args.dataset} dataset...")
        corpus_ids = list(test_corpus.keys())
        corpus_texts = [
            test_corpus[cid]["title"] + " " + test_corpus[cid]["text"]
            for cid in corpus_ids
        ]
        corpus_embeds = model.encode_document(corpus_texts, show_progress_bar=True)

        query_ids = list(test_queries.keys())
        query_texts = [test_queries[qid] for qid in query_ids]
        query_embeds = model.encode_query(query_texts, show_progress_bar=True)

        results = {}
        corpus_embeds_t = corpus_embeds.T
        for i in trange(len(query_embeds), desc="Scoring"):
            query_id = query_ids[i]
            query_embed = query_embeds[i].to_dense()
            scores = torch.matmul(query_embed, corpus_embeds_t)
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            ranked_doc_indices = scores.argsort()[::-1]
            results[query_id] = {
                corpus_ids[doc_idx]: float(scores[doc_idx])
                for doc_idx in ranked_doc_indices[:100]
            }

        k_values = [2, 10, 20, 100]
        ndc
        g, _map, recall, precision = EvaluateRetrieval.evaluate(
            test_qrels, results, k_values
        )

        recall_at_10 = recall["Recall@10"]
        recall_at_10_scores.append(recall_at_10)

        logging.info(f"Results for {num_samples} training samples:")
        logging.info(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
        for k in k_values:
            logging.info(f"Recall@{k}: {recall[f'Recall@{k}']:.4f}")

        # Save logs and model for this sample amount
        with open(os.path.join(sample_output_dir, "metrics.json"), "w") as f:
            # all the k values and their scores
            json.dump(
                {
                    "NDCG": ndcg,
                    "MAP": _map,
                    "Recall": recall,
                    "Precision": precision,
                },
                f,
                indent=4,
            )

        model.save_pretrained(sample_output_dir)
        logging.info(f"Model saved to {sample_output_dir}")

        if num_samples == training_amounts[-1]:
            final_model = model


if __name__ == "__main__":
    main()
