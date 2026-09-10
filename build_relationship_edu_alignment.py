import json
import time
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

MAX_WORKERS = 2
MAX_CONTEXTS = None

class RelationshipEduAlignment(BaseModel):
    segment_ids: list[str] = Field(default_factory=list)

def get_alignment_prompt() -> str:
    return '''
You are given a relationship between two entities and a list of identified text segments extracted from the passage associated with that relationship.

Relationship:
Source: {source}
Target: {target}
Description: {description}

Text segments:
{edus}

Select the smallest set of text segments that most directly corresponds to the textual evidence associated with the relationship.

Rules:
1. Use the relationship description only to understand which information should be located in the text.
2. The relationship may be expressed across multiple segments.
3. Evaluate selected segments together when they form a continuous or dependent piece of evidence.
4. Select only the minimum segments needed to locate the relationship in the passage.
5. Once the relationship can be identified from the selected segments, stop.
6. Include segments needed to resolve grammatical continuation, pronouns, references, headings, lists, or other discourse dependencies.
7. Do not select segments only because they mention the source or target if they are unrelated to the relationship.
8. Do not include additional background, examples, consequences, explanations, or other related information that is not necessary to locate the relationship.
9. If removing a selected segment would still leave enough textual information to locate the relationship, that segment must not be selected.
10. Do not try to verify, correct, expand, or rewrite the relationship. Your task is only to locate its corresponding textual evidence.
11. If none of the provided segments corresponds to the relationship, return an empty list.
12. Return only segment identifiers exactly as shown between brackets, such as SEG_000 or SEG_001.
13. Never return numbers that appear inside the text as segment identifiers.

Return the identifiers of the selected text segments.
'''

def load_jsonl(path: str | Path) -> list[dict]:
    result = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line:
                result.append(json.loads(line))

    return result

def append_jsonl(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
        file.flush()

def get_context_key(item: dict) -> tuple[str, str, str]:
    return (
        str(item['relationship_id']),
        str(item['document_id']),
        str(item['text_unit_id']),
    )

def load_processed_keys(path: str | Path) -> set[tuple[str, str, str]]:
    path = Path(path)

    if not path.exists():
        return set()

    result = set()

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                result.add(get_context_key(item))
            except (json.JSONDecodeError, KeyError):
                continue

    return result

def build_rst_index(rst_documents: list[dict]) -> dict[str, dict[int, dict]]:
    result = {}

    for document in rst_documents:
        document_id = str(document['document_id'])
        result[document_id] = {int(edu['edu_id']): edu for edu in document['edus']}

    return result

def load_llm(settings_path: str | Path) -> ChatOpenAI:
    with open(settings_path, 'r', encoding='utf-8') as file:
        settings = yaml.safe_load(file)

    config = settings['completion_models']['default_completion_model']

    return ChatOpenAI(
        model=config['model'],
        base_url=config.get('api_base'),
        api_key=config.get('api_key', 'local'),
        temperature=0,
    )

def get_candidate_edus(context: dict, rst_index: dict[str, dict[int, dict]]) -> list[dict]:
    document_id = str(context['document_id'])

    if document_id not in rst_index:
        raise ValueError(f'RST document not found: {document_id}')

    edu_index = rst_index[document_id]
    candidate_ids = [int(edu_id) for edu_id in context['candidate_edu_ids']]
    candidate_keys = context['candidate_edu_keys']

    if len(candidate_ids) != len(candidate_keys):
        raise ValueError('candidate_edu_ids and candidate_edu_keys have different lengths')

    result = []

    for edu_id, edu_key in zip(candidate_ids, candidate_keys):
        if edu_id not in edu_index:
            raise ValueError(f'EDU {edu_id} not found in document {document_id}')

        result.append({
            'edu_id': edu_id,
            'edu_key': edu_key,
            'text': edu_index[edu_id]['text'],
        })

    return result

def build_segment_index(edus: list[dict]) -> dict[str, dict]:
    return {
        f'SEG_{index:03d}': edu
        for index, edu in enumerate(edus)
    }

def normalize_segment_id(segment_id: str) -> str:
    match = re.fullmatch(r'SEG_(\d+)', segment_id.strip().upper())

    if not match:
        return segment_id

    return f'SEG_{int(match.group(1)):03d}'

def format_candidate_edus(segment_index: dict[str, dict]) -> str:
    return '\n'.join(
        f'[{segment_id}] {edu["text"]}'
        for segment_id, edu in segment_index.items()
    )

def align_context(context: dict, rst_index: dict[str, dict[int, dict]], chain) -> tuple[dict, float]:
    start = time.perf_counter()
    candidate_edus = get_candidate_edus(context, rst_index)
    segment_index = build_segment_index(candidate_edus)

    response = chain.invoke({
        'source': context.get('source', ''),
        'target': context.get('target', ''),
        'description': context.get('description', ''),
        'edus': format_candidate_edus(segment_index),
    })

    selected_segment_ids = list(dict.fromkeys(normalize_segment_id(segment_id) for segment_id in response.segment_ids))
    invalid_segment_ids = set(selected_segment_ids) - set(segment_index)

    if invalid_segment_ids:
        raise ValueError(
            f'Invalid segment IDs returned by LLM: {sorted(invalid_segment_ids)}'
        )

    selected_edus = sorted(
        (segment_index[segment_id] for segment_id in selected_segment_ids),
        key=lambda edu: edu['edu_id'],
    )

    selected_ids = [edu['edu_id'] for edu in selected_edus]

    result = {
        'relationship_id': str(context['relationship_id']),
        'source': context.get('source'),
        'target': context.get('target'),
        'description': context.get('description'),
        'document_id': str(context['document_id']),
        'text_unit_id': str(context['text_unit_id']),
        'candidate_edu_count': len(candidate_edus),
        'selected_edu_ids': selected_ids,
        'selected_edu_keys': [edu['edu_key'] for edu in selected_edus],
        'selected_edu_count': len(selected_ids),
        'selected_edus': selected_edus,
    }

    return result, time.perf_counter() - start

def format_time(seconds: float) -> str:
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def format_finish(seconds: float) -> str:
    finish = datetime.now() + timedelta(seconds=seconds)
    return finish.strftime('%d/%m %H:%M')

def main():
    base_path = Path(__file__).resolve().parent
    candidates_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_candidates.jsonl'
    rst_path = base_path / 'artifacts' / 'rst' / 'multihop_rst.jsonl'
    output_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_alignment.jsonl'
    error_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_alignment_errors.jsonl'
    settings_path = base_path / 'graphrag_workspace' / 'settings.yaml'

    contexts = load_jsonl(candidates_path)
    rst_documents = load_jsonl(rst_path)
    rst_index = build_rst_index(rst_documents)
    processed_keys = load_processed_keys(output_path)

    contexts_by_key = {}

    for context in contexts:
        key = get_context_key(context)

        if key not in contexts_by_key:
            contexts_by_key[key] = context

    unique_contexts = list(contexts_by_key.values())

    pending_by_key = {}

    for context in unique_contexts:
        key = get_context_key(context)

        if key not in processed_keys:
            pending_by_key[key] = context

    pending_contexts = list(pending_by_key.values())

    if MAX_CONTEXTS is not None:
        pending_contexts = pending_contexts[:MAX_CONTEXTS]

    llm = load_llm(settings_path)
    prompt = ChatPromptTemplate.from_template(get_alignment_prompt())
    structured_llm = llm.with_structured_output(RelationshipEduAlignment)
    chain = prompt | structured_llm

    total = len(unique_contexts)
    pending = len(pending_contexts)
    duplicate_contexts = len(contexts) - len(unique_contexts)

    print('Original contexts:', len(contexts))
    print('Unique contexts:', total)
    print('Duplicate contexts removed:', duplicate_contexts)
    print('Already processed:', len(processed_keys))
    print('Pending this run:', pending)
    print('Workers:', MAX_WORKERS)

    if pending == 0:
        print('All contexts already processed.')
        return

    completed_run = 0
    selected_contexts = 0
    empty_contexts = 0
    failed_contexts = 0
    processing_times = []
    selected_edu_total = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(align_context, context, rst_index, chain): context
            for context in pending_contexts
        }

        for future in as_completed(futures):
            context = futures[future]

            try:
                result, elapsed = future.result()
                append_jsonl(output_path, result)
                processed_keys.add(get_context_key(result))
                processing_times.append(elapsed)
                completed_run += 1
                selected_edu_total += result['selected_edu_count']

                if result['selected_edu_count'] > 0:
                    selected_contexts += 1
                else:
                    empty_contexts += 1

            except Exception as error:
                failed_contexts += 1

                append_jsonl(error_path, {
                    'relationship_id': str(context['relationship_id']),
                    'document_id': str(context['document_id']),
                    'text_unit_id': str(context['text_unit_id']),
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'timestamp': datetime.now().isoformat(),
                })

                print()
                print(f'ERROR | Relationship: {context["relationship_id"]} | TextUnit: {context["text_unit_id"]}')
                print(f'{type(error).__name__}: {error}')
                print()

            finished = completed_run + failed_contexts
            average_time = sum(processing_times) / len(processing_times) if processing_times else 0
            remaining = pending - finished
            eta = average_time * remaining / MAX_WORKERS if average_time > 0 else 0
            percentage = finished / pending * 100

            print(
                f'Alignment: {finished}/{pending} ({percentage:.2f}%) | '
                f'Selected: {selected_contexts} | Empty: {empty_contexts} | '
                f'Errors: {failed_contexts} | Avg: {format_time(average_time)} | '
                f'ETA: {format_time(eta)} | Finish: {format_finish(eta)}',
                flush=True,
            )

    average_selected = selected_edu_total / selected_contexts if selected_contexts > 0 else 0

    print()
    print('Relationship -> EDU semantic alignment finished')
    print('Unique contexts:', total)
    print('Processed this run:', completed_run)
    print('Contexts with selected EDUs:', selected_contexts)
    print('Contexts with no selected EDU:', empty_contexts)
    print('Errors:', failed_contexts)
    print('Selected EDUs:', selected_edu_total)
    print(f'Average selected EDUs per non-empty context: {average_selected:.2f}')
    print('Stored contexts:', len(processed_keys))
    print('Output:', output_path)

    if failed_contexts > 0:
        print('Errors:', error_path)

if __name__ == '__main__':
    main()