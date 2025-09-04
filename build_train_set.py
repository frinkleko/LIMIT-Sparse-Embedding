import json
import random
import os
import itertools
import pandas as pd
import requests
from tqdm import tqdm
import re


def get_names():
    """Downloads lists of first and last names."""
    csv_url = "https://gist.githubusercontent.com/craigh411/19a4479b289ae6c3f6edb95152214efc/raw/d25a1afd3de42f10abdea7740ed098d41de3c330/List%2520of%2520the%25201,000%2520Most%2520Common%2520Last%2520Names%2520(USA)"
    common_surnames_df = pd.read_csv(csv_url, names=["Surname", "None"])
    surname_list = common_surnames_df["Surname"].tolist()

    python_url = "https://gist.githubusercontent.com/ruanbekker/a1506f06aa1df06c5a9501cb393626ea/raw/cef847b6402da0fe00977e7349a4dc3fbeb4df54/array-names.py"
    response = requests.get(python_url)
    response.raise_for_status()
    python_content = response.text

    local_scope = {}
    exec(python_content, {}, local_scope)

    if "names" in local_scope:
        name_list = local_scope["names"]
    else:
        raise Exception(
            "Could not find 'names' variable in the downloaded Python content."
        )

    return list(set(name_list)), list(set(surname_list))


def generate_synthetic_dataset(
    output_dir,
    attributes,
    num_examples=1700,
    num_docs=50000,
    docs_per_query=2,
    attributes_per_doc_min=5,
    attributes_per_doc_max=15,
):
    """Generates and saves a synthetic LIMIT-style dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_names, unique_surnames = get_names()
    num_queries = num_examples // docs_per_query

    # 1. Generate Corpus
    corpus = {}
    doc_attributes = {}
    print("Generating corpus...")
    for i in tqdm(range(num_docs)):
        doc_id = f"doc_{i}"
        first_name = random.choice(unique_names)
        last_name = random.choice(unique_surnames)
        num_attributes = random.randint(attributes_per_doc_min, attributes_per_doc_max)
        assigned_attributes = random.sample(attributes, num_attributes)

        text = f"{first_name} {last_name} likes " + ", ".join(assigned_attributes) + "."
        corpus[doc_id] = {"_id": doc_id, "title": "", "text": text}
        doc_attributes[doc_id] = set(assigned_attributes)

    with open(os.path.join(output_dir, "corpus.jsonl"), "w") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + "\n")

    # 2. Generate Queries and Qrels
    queries = {}
    qrels = []

    attribute_to_docs = {}
    for doc_id, attrs in doc_attributes.items():
        for attr in attrs:
            if attr not in attribute_to_docs:
                attribute_to_docs[attr] = []
            attribute_to_docs[attr].append(doc_id)

    valid_attributes = [
        attr
        for attr, doc_ids in attribute_to_docs.items()
        if len(doc_ids) >= docs_per_query
    ]

    if len(valid_attributes) < num_queries:
        print(
            f"Warning: Only {len(valid_attributes)} attributes are available to generate {num_queries} queries. Some attributes will be reused."
        )
        query_attributes = random.choices(valid_attributes, k=num_queries)
    else:
        query_attributes = random.sample(valid_attributes, num_queries)

    print("Generating queries and qrels...")
    for i, attr in enumerate(tqdm(query_attributes)):
        query_id = f"query_{i}"
        queries[query_id] = {"_id": query_id, "text": f"Who likes {attr}?"}

        relevant_docs = attribute_to_docs[attr]
        positive_doc_ids = random.sample(relevant_docs, docs_per_query)

        for doc_id in positive_doc_ids:
            qrels.append({"query-id": query_id, "corpus-id": doc_id, "score": 1})

    with open(os.path.join(output_dir, "queries.jsonl"), "w") as f:
        for query in queries.values():
            f.write(json.dumps(query) + "\n")

    with open(os.path.join(output_dir, "qrels.jsonl"), "w") as f:
        for qrel in qrels:
            f.write(json.dumps(qrel) + "\n")

    print(f"Synthetic dataset with {len(qrels)} examples generated in {output_dir}")


def main():
    random.seed(42)
    # we just use the same notebook as in the original LIMIT dataset to get the master list of attributes
    with open("refs/enerate_limit_dataset.ipynb", "r") as f:
        notebook_content = json.load(f)

    items_to_like_str = ""
    for cell in notebook_content["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if 'items_to_like = """' in source:
                items_to_like_str = source.split('"""')[1].strip()
                break

    master_attributes = [
        item.strip() for item in items_to_like_str.split("\n") if item.strip()
    ]
    print(f"Found {len(master_attributes)} attributes in the master list.")

    test_attributes = set()
    with open("data/limit/queries.jsonl", "r") as f:
        for line in f:
            query_text = json.loads(line)["text"]
            attribute = re.sub(r"Who likes (.*)\?", r"\1", query_text).strip()
            test_attributes.add(attribute)
    print(f"Found {len(test_attributes)} attributes in the test set.")

    training_attributes = [
        attr for attr in master_attributes if attr not in test_attributes
    ]
    print(f"Created a pool of {len(training_attributes)} attributes for training.")

    output_dir = "data/limit-train"
    generate_synthetic_dataset(output_dir, training_attributes, num_examples=1700)


if __name__ == "__main__":
    main()
