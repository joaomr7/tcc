import argparse
import json
import sys

from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.rgrag_retriever import RGRAGRetriever

structure_results_path = project_root / 'artifacts' / 'evaluation' / 'rgrag_structure_results.jsonl'
results_path = project_root / 'artifacts' / 'evaluation' / 'dense_expansion_results.jsonl'
errors_path = project_root / 'artifacts' / 'evaluation' / 'dense_expansion_errors.jsonl'

workspace_path = project_root / 'graphrag_workspace'
rst_edges_path = project_root / 'artifacts' / 'rgrag' / 'rst_relationship_edges.jsonl'
consolidated_path = project_root / 'artifacts' / 'rgrag' / 'relationship_edu_consolidated.jsonl'


def normalize_text(text: str) -> str:
    return ''.join(text.split())


def get_text_unit_id(item: dict) -> str:
    return str(item.get('text_unit_id', item.get('id')))


def find_rank(fact: str, ranked: list[dict]) -> int | None:
    normalized_fact = normalize_text(fact)

    for rank, item in enumerate(ranked, start=1):
        if normalized_fact in normalize_text(item['text']):
            return rank

    return None


def ranking_metrics(ranks: list[int | None]) -> dict:
    gold_count = len(ranks)

    hits_4 = sum(
        rank is not None and rank <= 4
        for rank in ranks
    ) / gold_count

    hits_10 = sum(
        rank is not None and rank <= 10
        for rank in ranks
    ) / gold_count

    ranks_10 = sorted(
        rank
        for rank in ranks
        if rank is not None and rank <= 10
    )

    average_precision = sum(
        index / rank
        for index, rank in enumerate(ranks_10, start=1)
    ) / gold_count

    reciprocal_rank = (
        1 / ranks_10[0]
        if ranks_10
        else 0.0
    )

    complete_4 = float(
        all(
            rank is not None and rank <= 4
            for rank in ranks
        )
    )

    complete_10 = float(
        all(
            rank is not None and rank <= 10
            for rank in ranks
        )
    )

    return {
        'hits_at_4': hits_4,
        'hits_at_10': hits_10,
        'map_at_10': average_precision,
        'mrr_at_10': reciprocal_rank,
        'complete_at_4': complete_4,
        'complete_at_10': complete_10,
    }


def load_structure_results() -> list[dict]:
    results = []

    with open(structure_results_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))

    return results


def load_completed_indices() -> set[int]:
    if not results_path.exists():
        return set()

    completed = set()

    with open(results_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                completed.add(
                    json.loads(line)['dataset_index']
                )

    return completed


def evaluate_item(
    item: dict,
    retriever: RGRAGRetriever,
    all_text_unit_ids: set[str],
) -> dict:
    query = item['query']

    graphrag = retriever.graphrag.retrieve_candidates(query)

    graphrag_ids = {
        str(text_unit_id)
        for text_unit_id in graphrag['text_unit_ids']
    }

    if len(graphrag_ids) != item['graphrag_candidates']:
        raise RuntimeError(
            f'GraphRAG candidate mismatch: '
            f'expected {item["graphrag_candidates"]}, '
            f'found {len(graphrag_ids)}'
        )

    dense_ranked_all = retriever.ranker.rank(
        query=query,
        text_unit_ids=all_text_unit_ids,
        baseline_ids=all_text_unit_ids,
        rst_ids=set(),
        top_k=len(all_text_unit_ids),
    )

    budget = item['rst_new_candidates']

    dense_added_ids = []

    for ranked_item in dense_ranked_all:
        text_unit_id = get_text_unit_id(ranked_item)

        if text_unit_id in graphrag_ids:
            continue

        dense_added_ids.append(text_unit_id)

        if len(dense_added_ids) == budget:
            break

    dense_added_ids = set(dense_added_ids)

    candidate_ids = (
        graphrag_ids
        | dense_added_ids
    )

    if len(candidate_ids) != item['rgrag_candidates']:
        raise RuntimeError(
            f'Candidate budget mismatch: '
            f'expected {item["rgrag_candidates"]}, '
            f'found {len(candidate_ids)}'
        )

    ranked = [
        ranked_item
        for ranked_item in dense_ranked_all
        if get_text_unit_id(ranked_item) in candidate_ids
    ]

    evidence_results = []
    ranks = []

    for evidence in item['evidence']:
        rank = find_rank(
            evidence['fact'],
            ranked,
        )

        ranks.append(rank)

        evidence_results.append({
            'source': evidence['source'],
            'title': evidence['title'],
            'fact': evidence['fact'],
            'dense_expansion_candidate_found': rank is not None,
            'dense_expansion_rank': rank,
        })

    found = sum(
        rank is not None
        for rank in ranks
    )

    gold_count = len(ranks)

    top10 = []

    for rank, ranked_item in enumerate(
        ranked[:10],
        start=1,
    ):
        text_unit_id = get_text_unit_id(
            ranked_item
        )

        if text_unit_id in dense_added_ids:
            origin = 'dense'
        else:
            origin = 'graphrag'

        top10.append({
            'rank': rank,
            'text_unit_id': text_unit_id,
            'score': ranked_item['score'],
            'origin': origin,
            'text': ranked_item['text'],
        })

    return {
        'dataset_index': item['dataset_index'],
        'query': query,
        'question_type': item['question_type'],
        'gold_count': gold_count,
        'graphrag_candidates': len(graphrag_ids),
        'dense_new_candidates': len(dense_added_ids),
        'dense_expansion_candidates': len(candidate_ids),
        'rst_new_candidates_budget': budget,
        'dense_expansion_found': found,
        'dense_expansion_candidate_recall': found / gold_count,
        'dense_expansion_candidate_complete': float(
            found == gold_count
        ),
        'dense_expansion_metrics': ranking_metrics(ranks),
        'evidence': evidence_results,
        'dense_expansion_top10': top10,
    }


def main(limit: int | None = None):
    dataset = load_structure_results()

    if limit is not None:
        dataset = dataset[:limit]

    retriever = RGRAGRetriever(
        workspace_path=workspace_path,
        rst_edges_path=rst_edges_path,
        consolidated_path=consolidated_path,
        community_level=2,
    )

    all_text_unit_ids = {
        str(text_unit.id)
        for text_unit in retriever.graphrag.text_units
    }

    completed_indices = load_completed_indices()

    print('Queries:', len(dataset))
    print('TextUnits:', len(all_text_unit_ids))
    print('Completed:', len(completed_indices))

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        results_path,
        'a',
        encoding='utf-8',
    ) as results_file, open(
        errors_path,
        'a',
        encoding='utf-8',
    ) as errors_file:
        for item in tqdm(dataset):
            dataset_index = item['dataset_index']

            if dataset_index in completed_indices:
                continue

            try:
                result = evaluate_item(
                    item=item,
                    retriever=retriever,
                    all_text_unit_ids=all_text_unit_ids,
                )

                results_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + '\n'
                )

                results_file.flush()

            except Exception as error:
                errors_file.write(
                    json.dumps(
                        {
                            'dataset_index': dataset_index,
                            'query': item['query'],
                            'error': str(error),
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )

                errors_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
    )