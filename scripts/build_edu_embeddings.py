import argparse
import json
import sys

from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.graphrag_retriever import GraphRAGRetriever

rst_path = project_root / 'artifacts' / 'rst' / 'multihop_rst.jsonl'
embeddings_path = project_root / 'artifacts' / 'rgrag' / 'edu_embeddings.npy'
index_path = project_root / 'artifacts' / 'rgrag' / 'edu_index.json'

def load_all_edus() -> dict[str, str]:
    edus = {}

    with open(rst_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            for edu in item['edus']:
                edus[edu['edu_key']] = edu['text']

    return edus

def main(batch_size: int):
    all_edus = load_all_edus()

    print('Total EDUs:', len(all_edus))

    if embeddings_path.exists() and index_path.exists():
        existing_vectors = np.load(embeddings_path)

        with open(index_path, 'r', encoding='utf-8') as file:
            edu_index = json.load(file)

        print('Existing embeddings:', len(edu_index))
    else:
        existing_vectors = np.empty((0, 1024), dtype=np.float32)
        edu_index = {}

    missing_keys = [
        edu_key
        for edu_key in all_edus
        if edu_key not in edu_index
    ]

    print('Missing embeddings:', len(missing_keys))

    if not missing_keys:
        print('All EDU embeddings already exist.')
        return

    print()
    print('Initializing GraphRAG embedder...')

    graphrag = GraphRAGRetriever(
        workspace_path=project_root / 'graphrag_workspace',
        community_level=2,
    )

    embedder = graphrag.search_engine.context_builder.text_embedder

    print('Embedder initialized.')
    print()

    new_vectors = []

    for start in tqdm(range(0, len(missing_keys), batch_size), desc='Embedding EDUs'):
        batch_keys = missing_keys[start:start + batch_size]
        batch_texts = [all_edus[key] for key in batch_keys]

        response = embedder.embedding(input=batch_texts)

        batch_vectors = np.asarray(
            response.embeddings,
            dtype=np.float32,
        )

        norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
        batch_vectors = batch_vectors / np.clip(norms, 1e-12, None)

        new_vectors.append(batch_vectors)

    new_vectors = np.concatenate(new_vectors, axis=0)

    start_index = len(existing_vectors)

    for offset, edu_key in enumerate(missing_keys):
        edu_index[edu_key] = start_index + offset

    vectors = np.concatenate(
        [existing_vectors, new_vectors],
        axis=0,
    )

    np.save(embeddings_path, vectors)

    with open(index_path, 'w', encoding='utf-8') as file:
        json.dump(edu_index, file, ensure_ascii=False)

    print()
    print('Final EDUs:', len(edu_index))
    print('Embedding shape:', vectors.shape)
    print('Embedding dtype:', vectors.dtype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    main(batch_size=args.batch_size)