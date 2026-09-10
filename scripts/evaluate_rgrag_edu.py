import argparse
import json
import sys

from pathlib import Path

from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.rgrag_edu_retriever import RGRAGEDURetriever

results_path = project_root / 'artifacts' / 'evaluation' / 'rgrag_edu_results.jsonl'
errors_path = project_root / 'artifacts' / 'evaluation' / 'rgrag_edu_errors.jsonl'

def normalize_text(text: str) -> str:
    return text.replace(' ', '').replace('\n', '')

def contains_fact(text: str, fact: str) -> bool:
    return normalize_text(fact) in normalize_text(text)

def ranking_metrics(ranked: list[dict], evidence_list: list[dict], k: int = 10) -> dict:
    retrieved = [normalize_text(item['text']) for item in ranked[:k]]
    gold = [normalize_text(item['fact']) for item in evidence_list]

    found_gold = set()
    found_at_4 = set()
    average_precision_sum = 0
    first_relevant_rank = None

    for rank, text in enumerate(retrieved, start=1):
        count = 0

        for gold_index, fact in enumerate(gold):
            if fact not in text:
                continue

            if rank <= 4:
                found_at_4.add(gold_index)

            if gold_index in found_gold:
                continue

            found_gold.add(gold_index)
            count += 1

        if count == 0:
            continue

        if first_relevant_rank is None:
            first_relevant_rank = rank

        average_precision_sum += count / rank

    return {
        'hits_at_4': int(len(found_at_4) > 0),
        'hits_at_10': int(len(found_gold) > 0),
        'map_at_10': average_precision_sum / min(len(gold), 10),
        'mrr_at_10': 1 / first_relevant_rank if first_relevant_rank else 0,
        'complete_at_4': int(len(found_at_4) == len(gold)),
        'complete_at_10': int(len(found_gold) == len(gold)),
    }

def find_rank(fact: str, ranked: list[dict]):
    for rank, item in enumerate(ranked, start=1):
        if contains_fact(item['text'], fact):
            return rank, item

    return None, None

def evaluate_item(retriever, dataset_index: int, item: dict) -> dict:
    query = item['query']
    evidence_list = item['evidence_list']

    candidates = retriever.rgrag.retrieve_candidates(query)

    graphrag_ids = {
        str(text_unit_id)
        for text_unit_id in candidates['graphrag']['text_unit_ids']
    }

    rst_ids = {
        str(text_unit_id)
        for text_unit_id in candidates['rst']['text_unit_ids']
    }

    rgrag_ids = graphrag_ids | rst_ids

    ranked = retriever.ranker.rank(
        query=query,
        text_unit_ids=rgrag_ids,
        graphrag_ids=graphrag_ids,
        rst_ids=rst_ids,
        top_k=len(rgrag_ids),
    )

    evidence_results = []

    for evidence in evidence_list:
        rank, ranked_item = find_rank(
            evidence['fact'],
            ranked,
        )

        evidence_results.append({
            'source': evidence.get('source'),
            'title': evidence.get('title'),
            'fact': evidence['fact'],
            'candidate_found': rank is not None,
            'rank': rank,
            'origin': ranked_item['origin'] if ranked_item else None,
            'score': ranked_item['score'] if ranked_item else None,
            'score_source': ranked_item['score_source'] if ranked_item else None,
            'text_unit_score': ranked_item['text_unit_score'] if ranked_item else None,
            'edu_score': ranked_item['edu_score'] if ranked_item else None,
            'best_edu_key': ranked_item['best_edu_key'] if ranked_item else None,
            'best_edu_text': ranked_item['best_edu_text'] if ranked_item else None,
        })

    gold_count = len(evidence_results)
    gold_found = sum(evidence['candidate_found'] for evidence in evidence_results)

    edu_covered_candidates = len(
        set(retriever.ranker.text_unit_edu_rows)
        & rgrag_ids
    )

    return {
        'dataset_index': dataset_index,
        'query': query,
        'question_type': item['question_type'],
        'gold_count': gold_count,
        'graphrag_candidates': len(graphrag_ids),
        'rst_new_candidates': len(rst_ids - graphrag_ids),
        'rgrag_candidates': len(rgrag_ids),
        'edu_covered_candidates': edu_covered_candidates,
        'candidate_recall': gold_found / gold_count,
        'candidate_complete': gold_found == gold_count,
        'metrics': ranking_metrics(ranked, evidence_list),
        'evidence': evidence_results,
        'top10': [
            {
                'text_unit_id': ranked_item['text_unit_id'],
                'score': ranked_item['score'],
                'origin': ranked_item['origin'],
                'score_source': ranked_item['score_source'],
                'text_unit_score': ranked_item['text_unit_score'],
                'edu_score': ranked_item['edu_score'],
                'best_edu_key': ranked_item['best_edu_key'],
                'best_edu_text': ranked_item['best_edu_text'],
            }
            for ranked_item in ranked[:10]
        ],
    }

def load_completed() -> set[int]:
    completed = set()

    if not results_path.exists():
        return completed

    with open(results_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line:
                completed.add(json.loads(line)['dataset_index'])

    return completed

def find_dataset() -> Path:
    paths = list(project_root.rglob('MultiHopRAG.json'))

    if not paths:
        raise FileNotFoundError('MultiHopRAG.json not found.')

    return paths[0]

def main(limit: int | None = None):
    results_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()

    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    valid_dataset = [
        (index, item)
        for index, item in enumerate(dataset)
        if item['question_type'] != 'null_query' and item['evidence_list']
    ]

    completed = load_completed()

    remaining = [
        (index, item)
        for index, item in valid_dataset
        if index not in completed
    ]

    if limit is not None:
        remaining = remaining[:limit]

    print('Dataset:', dataset_path)
    print('Valid queries:', len(valid_dataset))
    print('Already completed:', len(completed))
    print('Queries in this run:', len(remaining))

    if not remaining:
        print('No remaining queries.')
        return

    print()
    print('Initializing RGRAG-EDU...')

    retriever = RGRAGEDURetriever(
        workspace_path=project_root / 'graphrag_workspace',
        rst_edges_path=project_root / 'artifacts' / 'rgrag' / 'rst_relationship_edges.jsonl',
        consolidated_path=project_root / 'artifacts' / 'rgrag' / 'relationship_edu_consolidated.jsonl',
        rst_path=project_root / 'artifacts' / 'rst' / 'multihop_rst.jsonl',
        embeddings_path=project_root / 'artifacts' / 'rgrag' / 'edu_embeddings.npy',
        index_path=project_root / 'artifacts' / 'rgrag' / 'edu_index.json',
        community_level=2,
    )

    print('RGRAG-EDU initialized.')
    print('Mapped TextUnits:', len(retriever.ranker.text_unit_edu_rows))
    print()

    success = 0
    errors = 0

    with open(results_path, 'a', encoding='utf-8') as results_file, open(errors_path, 'a', encoding='utf-8') as errors_file:
        for dataset_index, item in tqdm(remaining, desc='RGRAG-EDU'):
            try:
                result = evaluate_item(
                    retriever,
                    dataset_index,
                    item,
                )

                results_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    ) + '\n'
                )
                results_file.flush()

                success += 1

            except Exception as error:
                errors_file.write(
                    json.dumps({
                        'dataset_index': dataset_index,
                        'query': item['query'],
                        'error': repr(error),
                    }, ensure_ascii=False) + '\n'
                )
                errors_file.flush()

                errors += 1

    print()
    print('Completed in this run:', success)
    print('Errors in this run:', errors)
    print('Results:', results_path)
    print('Errors:', errors_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    main(limit=args.limit)