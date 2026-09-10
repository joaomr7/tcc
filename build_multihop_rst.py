import hashlib
import json
import time

from datetime import datetime, timedelta
from pathlib import Path

from src.rst import DMRSTParser

DMRST_BATCH_SIZE = 2

def get_document_id(item: dict) -> str:
    value = '|'.join([
        str(item.get('title', '')),
        str(item.get('source', '')),
        str(item.get('published_at', '')),
    ])
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def load_corpus(path: str | Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def build_edu_spans(document_id: str, document: str, rst: dict, tokenizer) -> list[dict]:
    encoding = tokenizer(document, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = encoding['offset_mapping']
    tokens = rst['tokens']
    edu_breaks = rst['edu_breaks']
    parser_edus = rst['edus']

    if len(offsets) != len(tokens):
        raise ValueError(f'Token count mismatch: tokenizer={len(offsets)}, DMRST={len(tokens)}')

    result = []
    start_token = 0

    for edu_id, end_token in enumerate(edu_breaks):
        if end_token >= len(offsets):
            raise ValueError(f'Invalid EDU break {end_token} for {len(offsets)} tokens')

        start = offsets[start_token][0]
        end = offsets[end_token][1]

        result.append({
            'edu_id': edu_id,
            'edu_key': f'{document_id}:{edu_id}',
            'text': document[start:end],
            'parser_text': parser_edus[edu_id],
            'start': start,
            'end': end,
        })

        start_token = end_token + 1

    return result

def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    processed_ids = set()

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                processed_ids.add(json.loads(line)['document_id'])
            except json.JSONDecodeError:
                continue

    return processed_ids

def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
        file.flush()

def format_time(seconds: float | None) -> str:
    if seconds is None:
        return 'calculando...'

    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def format_finish(seconds: float | None) -> str:
    if seconds is None:
        return 'calculando...'

    finish = datetime.now() + timedelta(seconds=seconds)
    return finish.strftime('%d/%m %H:%M')

def main():
    base_path = Path(__file__).resolve().parent

    corpus_path = base_path / 'artifacts' / 'dataset' / 'corpus.json'
    output_path = base_path / 'artifacts' / 'rst' / 'multihop_rst.jsonl'
    error_path = base_path / 'artifacts' / 'rst' / 'multihop_rst_errors.jsonl'

    dmrst_checkpoint_path = base_path / 'artifacts' / 'dmrst' / 'multi_all_checkpoint.torchsave'
    dmrst_cache_dir = base_path / 'artifacts' / 'models'

    corpus = load_corpus(corpus_path)
    processed_ids = load_processed_ids(output_path)

    dmrst = DMRSTParser(dmrst_checkpoint_path, dmrst_cache_dir, DMRST_BATCH_SIZE)

    total = len(corpus)

    print('Total documents:', total)
    print('Already processed:', len(processed_ids))
    print('Remaining:', total - len(processed_ids))

    if len(processed_ids) == total:
        print('All documents already processed.')
        return

    execution_start = time.perf_counter()
    processing_times = []
    processed_this_run = 0
    failed_this_run = 0

    for corpus_index, item in enumerate(corpus):
        document_id = get_document_id(item)

        if document_id in processed_ids:
            continue

        document = item.get('body', '')

        if not document.strip():
            append_jsonl(error_path, {
                'document_id': document_id,
                'corpus_index': corpus_index,
                'title': item.get('title'),
                'error': 'Empty document',
                'timestamp': datetime.now().isoformat(),
            })
            failed_this_run += 1
            continue

        document_start = time.perf_counter()

        try:
            rst = dmrst.parse(document)

            if not isinstance(rst, dict) or 'edus' not in rst or 'tree' not in rst or 'tokens' not in rst or 'edu_breaks' not in rst:
                raise ValueError('Invalid DMRST response')

            edus = build_edu_spans(document_id, document, rst, dmrst.tokenizer)

            result = {
                'document_id': document_id,
                'corpus_index': corpus_index,
                'title': item.get('title'),
                'source': item.get('source'),
                'published_at': item.get('published_at'),
                'document_length': len(document),
                'edu_count': len(edus),
                'edus': edus,
                'tree': rst['tree'],
            }

            append_jsonl(output_path, result)
            processed_ids.add(document_id)
            processed_this_run += 1

        except Exception as error:
            append_jsonl(error_path, {
                'document_id': document_id,
                'corpus_index': corpus_index,
                'title': item.get('title'),
                'error_type': type(error).__name__,
                'error': str(error),
                'timestamp': datetime.now().isoformat(),
            })

            failed_this_run += 1

            print(f'ERROR | {corpus_index + 1}/{total} | {item.get("title")}')
            print(f'{type(error).__name__}: {error}')
            continue

        document_elapsed = time.perf_counter() - document_start
        processing_times.append(document_elapsed)

        average_time = sum(processing_times) / len(processing_times)
        completed = len(processed_ids)
        remaining = total - completed
        eta = average_time * remaining
        percentage = completed / total * 100

        print(
            f'RST: {completed}/{total} ({percentage:.2f}%) | '
            f'EDUs: {len(edus)} | '
            f'Doc: {format_time(document_elapsed)} | '
            f'Média: {format_time(average_time)} | '
            f'ETA: {format_time(eta)} | '
            f'Término: {format_finish(eta)}'
        )

    execution_elapsed = time.perf_counter() - execution_start

    print()
    print('RST processing finished')
    print('Successfully stored:', len(processed_ids))
    print('Processed this run:', processed_this_run)
    print('Failed this run:', failed_this_run)
    print('Execution time:', format_time(execution_elapsed))
    print('Output:', output_path)

    if failed_this_run > 0:
        print('Errors:', error_path)

if __name__ == '__main__':
    main()