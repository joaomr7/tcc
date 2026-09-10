import ast
import json

from pathlib import Path

import pandas as pd

def load_jsonl(path: str | Path) -> list[dict]:
    result = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line:
                result.append(json.loads(line))

    return result

def save_jsonl(path: str | Path, data: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def parse_id_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        if value.startswith('['):
            try:
                parsed = ast.literal_eval(value)
                return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                pass

        return [value]

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    if hasattr(value, 'tolist'):
        parsed = value.tolist()

        if isinstance(parsed, list):
            return [str(item) for item in parsed]

        return [str(parsed)]

    return [str(value)]

def build_alignment_index(alignments: list[dict]) -> dict[str, dict]:
    result = {}

    for alignment in alignments:
        text_unit_id = str(alignment['text_unit_id'])

        if text_unit_id in result:
            raise ValueError(f'Duplicate TextUnit alignment: {text_unit_id}')

        result[text_unit_id] = alignment

    return result

def main():
    base_path = Path(__file__).resolve().parent
    relationships_path = base_path / 'graphrag_workspace' / 'output' / 'relationships.parquet'
    alignment_path = base_path / 'artifacts' / 'rgrag' / 'text_unit_edu_alignment.jsonl'
    output_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_candidates.jsonl'
    error_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_candidates_errors.jsonl'

    relationships = pd.read_parquet(relationships_path)
    alignments = load_jsonl(alignment_path)
    alignment_by_text_unit = build_alignment_index(alignments)

    results = []
    errors = []
    seen_contexts = set()
    relationships_with_candidates = set()
    relationships_multiple_text_units = 0
    duplicate_text_unit_ids = 0
    duplicate_contexts = 0

    total = len(relationships)

    for position, (_, row) in enumerate(relationships.iterrows(), start=1):
        relationship = row.to_dict()
        relationship_id = str(relationship['id'])

        original_text_unit_ids = parse_id_list(relationship.get('text_unit_ids'))
        text_unit_ids = list(dict.fromkeys(original_text_unit_ids))

        duplicate_text_unit_ids += len(original_text_unit_ids) - len(text_unit_ids)

        if len(text_unit_ids) > 1:
            relationships_multiple_text_units += 1

        relationship_has_candidates = False

        for text_unit_id in text_unit_ids:
            alignment = alignment_by_text_unit.get(text_unit_id)

            if alignment is None:
                errors.append({
                    'relationship_id': relationship_id,
                    'text_unit_id': text_unit_id,
                    'error': 'TextUnit alignment not found',
                })
                continue

            document_id = str(alignment['document_id'])
            context_key = (relationship_id, document_id, text_unit_id)

            if context_key in seen_contexts:
                duplicate_contexts += 1
                continue

            candidate_edu_ids = alignment['edu_ids']
            candidate_edu_keys = alignment['edu_keys']

            if not candidate_edu_ids:
                errors.append({
                    'relationship_id': relationship_id,
                    'text_unit_id': text_unit_id,
                    'document_id': document_id,
                    'error': 'TextUnit has no candidate EDUs',
                })
                continue

            results.append({
                'relationship_id': relationship_id,
                'source': relationship.get('source'),
                'target': relationship.get('target'),
                'description': relationship.get('description'),
                'document_id': document_id,
                'text_unit_id': text_unit_id,
                'candidate_edu_ids': candidate_edu_ids,
                'candidate_edu_keys': candidate_edu_keys,
                'candidate_edu_count': len(candidate_edu_ids),
            })

            seen_contexts.add(context_key)
            relationship_has_candidates = True

        if relationship_has_candidates:
            relationships_with_candidates.add(relationship_id)

        percentage = position / total * 100

        print(
            f'Candidates: {position}/{total} ({percentage:.2f}%) | '
            f'Contexts: {len(results)} | Errors: {len(errors)}',
            flush=True,
        )

    save_jsonl(output_path, results)

    if errors:
        save_jsonl(error_path, errors)
    elif error_path.exists():
        error_path.unlink()

    candidate_counts = [item['candidate_edu_count'] for item in results]
    average_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
    max_candidates = max(candidate_counts) if candidate_counts else 0

    print()
    print('Relationship -> TextUnit -> EDU candidate generation finished')
    print('Relationships:', total)
    print('Relationships with candidates:', len(relationships_with_candidates))
    print('Relationship-TextUnit contexts:', len(results))
    print('Relationships with multiple TextUnits:', relationships_multiple_text_units)
    print('Duplicate TextUnit IDs removed:', duplicate_text_unit_ids)
    print('Duplicate contexts removed:', duplicate_contexts)
    print('Errors:', len(errors))
    print(f'Average candidate EDUs per context: {average_candidates:.2f}')
    print('Maximum candidate EDUs in a context:', max_candidates)
    print('Output:', output_path)

    if errors:
        print('Errors:', error_path)

if __name__ == '__main__':
    main()