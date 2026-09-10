import ast
import bisect
import json
import unicodedata

from collections import defaultdict
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

def normalize_text_with_mapping(text: str) -> tuple[str, list[int]]:
    normalized_chars = []
    original_positions = []
    previous_space = False

    for index, char in enumerate(text):
        normalized = unicodedata.normalize('NFKC', char)

        for normalized_char in normalized:
            if normalized_char.isspace():
                if normalized_chars and not previous_space:
                    normalized_chars.append(' ')
                    original_positions.append(index)

                previous_space = True
                continue

            normalized_chars.append(normalized_char)
            original_positions.append(index)
            previous_space = False

    return ''.join(normalized_chars), original_positions

def normalize_text(text: str) -> str:
    normalized, _ = normalize_text_with_mapping(text)
    return normalized.strip()

def find_text_unit_span(document: str, text_unit_text: str, search_start: int = 0) -> tuple[int, int]:
    position = document.find(text_unit_text, search_start)

    if position >= 0:
        return position, position + len(text_unit_text)

    stripped_text = text_unit_text.strip()

    if stripped_text:
        position = document.find(stripped_text, search_start)

        if position >= 0:
            return position, position + len(stripped_text)

    normalized_document, mapping = normalize_text_with_mapping(document)
    normalized_text_unit = normalize_text(text_unit_text)

    if not normalized_text_unit:
        raise ValueError('TextUnit is empty after normalization')

    normalized_search_start = bisect.bisect_left(mapping, search_start)
    position = normalized_document.find(normalized_text_unit, normalized_search_start)

    if position < 0:
        position = normalized_document.find(normalized_text_unit)

    if position < 0:
        raise ValueError(f'Could not locate TextUnit in document: {text_unit_text[:150]!r}')

    normalized_end = position + len(normalized_text_unit) - 1
    start = mapping[position]
    end = mapping[normalized_end] + 1

    return start, end

def get_document_ids(text_unit: dict) -> list[str]:
    if 'document_ids' in text_unit:
        return parse_id_list(text_unit['document_ids'])

    if 'document_id' in text_unit:
        return parse_id_list(text_unit['document_id'])

    raise ValueError('TextUnit does not contain document_ids or document_id')

def text_unit_order(text_unit: dict):
    value = text_unit.get('human_readable_id')

    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)

def get_edu_matches(text_unit_start: int, text_unit_end: int, edus: list[dict]) -> list[dict]:
    matches = []

    for edu in edus:
        edu_start = edu.get('start')
        edu_end = edu.get('end')

        if edu_start is None or edu_end is None:
            continue

        overlap_start = max(text_unit_start, edu_start)
        overlap_end = min(text_unit_end, edu_end)
        overlap_chars = overlap_end - overlap_start

        if overlap_chars <= 0:
            continue

        matches.append({
            'edu_id': edu['edu_id'],
            'edu_key': edu['edu_key'],
            'edu_start': edu_start,
            'edu_end': edu_end,
            'overlap_chars': overlap_chars,
        })

    return matches

def main():
    base_path = Path(__file__).resolve().parent
    graphrag_output_path = base_path / 'graphrag_workspace' / 'output'
    documents_path = graphrag_output_path / 'documents.parquet'
    text_units_path = graphrag_output_path / 'text_units.parquet'
    rst_path = base_path / 'artifacts' / 'rst' / 'multihop_rst.jsonl'
    output_path = base_path / 'artifacts' / 'rgrag' / 'text_unit_edu_alignment.jsonl'
    error_path = base_path / 'artifacts' / 'rgrag' / 'text_unit_edu_alignment_errors.jsonl'

    documents = pd.read_parquet(documents_path)
    text_units = pd.read_parquet(text_units_path)
    rst_documents = load_jsonl(rst_path)

    documents_by_id = {str(row['id']): row['text'] for _, row in documents.iterrows()}
    rst_by_document = {str(item['document_id']): item for item in rst_documents}

    text_units_by_document = defaultdict(list)

    for _, row in text_units.iterrows():
        text_unit = row.to_dict()
        document_ids = get_document_ids(text_unit)

        if len(document_ids) != 1:
            raise ValueError(f'TextUnit {text_unit["id"]} has {len(document_ids)} documents: {document_ids}')

        text_units_by_document[document_ids[0]].append(text_unit)

    alignments = []
    errors = []
    aligned_edu_keys = set()

    total_text_units = len(text_units)
    processed = 0

    for document_id, document_text_units in text_units_by_document.items():
        if document_id not in documents_by_id:
            errors.append({
                'document_id': document_id,
                'error': 'Document not found in documents.parquet',
            })
            continue

        if document_id not in rst_by_document:
            errors.append({
                'document_id': document_id,
                'error': 'Document not found in RST output',
            })
            continue

        document = documents_by_id[document_id]
        rst = rst_by_document[document_id]
        edus = rst['edus']
        document_text_units = sorted(document_text_units, key=text_unit_order)

        previous_start = -1

        for text_unit in document_text_units:
            text_unit_id = str(text_unit['id'])
            text_unit_text = text_unit['text']

            try:
                start, end = find_text_unit_span(document, text_unit_text, previous_start + 1)
                edu_matches = get_edu_matches(start, end, edus)

                if not edu_matches:
                    raise ValueError('TextUnit does not overlap any EDU')

                edu_ids = [item['edu_id'] for item in edu_matches]
                edu_keys = [item['edu_key'] for item in edu_matches]

                alignments.append({
                    'text_unit_id': text_unit_id,
                    'document_id': document_id,
                    'text_unit_start': start,
                    'text_unit_end': end,
                    'text_unit_length': end - start,
                    'edu_ids': edu_ids,
                    'edu_keys': edu_keys,
                    'edu_matches': edu_matches,
                })

                aligned_edu_keys.update(edu_keys)
                previous_start = start

            except Exception as error:
                errors.append({
                    'text_unit_id': text_unit_id,
                    'document_id': document_id,
                    'error_type': type(error).__name__,
                    'error': str(error),
                })

            processed += 1
            percentage = processed / total_text_units * 100

            print(
                f'Alignment: {processed}/{total_text_units} ({percentage:.2f}%) | '
                f'Aligned: {len(alignments)} | Errors: {len(errors)}',
                flush=True,
            )

    total_edus = sum(len(item['edus']) for item in rst_documents)
    edu_coverage = len(aligned_edu_keys) / total_edus * 100 if total_edus > 0 else 0

    save_jsonl(output_path, alignments)

    if errors:
        save_jsonl(error_path, errors)
    elif error_path.exists():
        error_path.unlink()

    print()
    print('RGRAG TextUnit <-> EDU alignment finished')
    print('Documents:', len(documents))
    print('RST documents:', len(rst_documents))
    print('TextUnits:', total_text_units)
    print('Aligned TextUnits:', len(alignments))
    print('Errors:', len(errors))
    print('Total EDUs:', total_edus)
    print('EDUs covered by TextUnits:', len(aligned_edu_keys))
    print(f'EDU coverage: {edu_coverage:.2f}%')
    print('Output:', output_path)

    if errors:
        print('Errors:', error_path)

if __name__ == '__main__':
    main()