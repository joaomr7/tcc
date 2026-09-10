import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import graphrag.api as api

from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.load_config import load_config

def get_document_id(item: dict) -> str:
    value = '|'.join([
        str(item.get('title', '')),
        str(item.get('source', '')),
        str(item.get('published_at', '')),
    ])

    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def load_multihop_corpus(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    with open(path, 'r', encoding='utf-8') as file:
        corpus = json.load(file)

    documents = []

    for item in corpus:
        documents.append({
            'id': get_document_id(item),
            'title': item['title'],
            'text': item['body'],
            'creation_date': item.get('published_at'),
            'raw_data': item,
        })

    return pd.DataFrame(documents)

def format_time(seconds: float | None) -> str:
    if seconds is None:
        return 'calculando...'

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def format_finish(seconds: float | None) -> str:
    if seconds is None:
        return 'calculando...'

    finish = datetime.now() + timedelta(seconds=seconds)

    return finish.strftime('%d/%m %H:%M')

class ProgressCallback(NoopWorkflowCallbacks):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.pipeline_start_time = None
        self.workflow = None
        self.workflow_start_time = None

    def pipeline_start(self, names: list[str]) -> None:
        self.pipeline_start_time = time.perf_counter()

        print(
            f'Pipeline iniciado | '
            f'Workflows: {len(names)} | '
            f'ETA: calculando...'
        )

    def workflow_start(self, name: str, instance: object) -> None:
        self.workflow = name
        self.workflow_start_time = time.perf_counter()

        print()
        print(
            f'{name}: iniciado | '
            f'ETA: calculando...'
        )

    def progress(self, progress) -> None:
        if progress.total_items is None or progress.completed_items is None:
            return

        total = progress.total_items
        completed = progress.completed_items

        if total == 0:
            return

        elapsed = time.perf_counter() - self.workflow_start_time

        eta = None

        if completed > 0:
            seconds_per_item = elapsed / completed
            eta = seconds_per_item * (total - completed)

        percentage = completed / total * 100

        pipeline_elapsed = (
            time.perf_counter() - self.pipeline_start_time
            if self.pipeline_start_time is not None
            else 0
        )

        data = {
            'workflow': self.workflow,
            'description': progress.description,
            'completed': completed,
            'total': total,
            'percentage': percentage,
            'workflow_elapsed_seconds': elapsed,
            'workflow_eta_seconds': eta,
            'pipeline_elapsed_seconds': pipeline_elapsed,
            'estimated_finish': format_finish(eta),
            'updated_at': datetime.now().isoformat(),
        }

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )

        print(
            f'{self.workflow}: '
            f'{completed}/{total} '
            f'({percentage:.2f}%) | '
            f'Decorrido: {format_time(elapsed)} | '
            f'ETA: {format_time(eta)} | '
            f'Término: {format_finish(eta)}',
            flush=True,
        )

    def workflow_end(self, name: str, instance: object) -> None:
        elapsed = (
            time.perf_counter() - self.workflow_start_time
            if self.workflow_start_time is not None
            else 0
        )

        print(
            f'{name}: concluído | '
            f'Tempo: {format_time(elapsed)} | '
            f'ETA: 00:00:00'
        )

    def pipeline_error(self, error: BaseException) -> None:
        elapsed = (
            time.perf_counter() - self.pipeline_start_time
            if self.pipeline_start_time is not None
            else 0
        )

        print()
        print(
            f'Pipeline ERROR: {error} | '
            f'Decorrido: {format_time(elapsed)} | '
            f'ETA: indisponível'
        )

    def pipeline_end(self, results) -> None:
        elapsed = (
            time.perf_counter() - self.pipeline_start_time
            if self.pipeline_start_time is not None
            else 0
        )

        print()
        print(
            f'Pipeline concluído | '
            f'Tempo total: {format_time(elapsed)} | '
            f'ETA: 00:00:00'
        )


async def main():
    base_path = Path(__file__).resolve().parent

    root = base_path / 'graphrag_workspace'
    corpus_path = base_path / 'artifacts' / 'dataset' / 'corpus.json'
    progress_path = base_path / 'artifacts' / 'graphrag_progress.json'
    text_units_path = root / 'output' / 'text_units.parquet'

    config = load_config(root)

    documents = load_multihop_corpus(corpus_path)

    callback = ProgressCallback(progress_path)

    print(
        f'Documents: {len(documents)} | '
        f'ETA: calculando...'
    )

    print()

    start = time.perf_counter()

    results = await api.build_index(
        config=config,
        input_documents=documents,
        callbacks=[callback],
    )

    elapsed = time.perf_counter() - start

    print()

    for result in results:
        if result.error:
            print(
                f'{result.workflow}: ERROR {result.error} | '
                f'Decorrido total: {format_time(elapsed)} | '
                f'ETA: indisponível'
            )
        else:
            print(
                f'{result.workflow}: OK | '
                f'Decorrido total: {format_time(elapsed)} | '
                f'ETA: 00:00:00'
            )

    if text_units_path.exists():
        text_units = pd.read_parquet(text_units_path)

        print()
        print(
            f'Chunks: {len(text_units)} | '
            f'Tempo total: {format_time(elapsed)} | '
            f'ETA: 00:00:00'
        )

        if len(text_units) > 0:
            print(
                f'Segundos por chunk: '
                f'{elapsed / len(text_units):.2f} | '
                f'ETA: 00:00:00'
            )

    print()
    print(
        f'Execution time: {format_time(elapsed)} | '
        f'Total seconds: {elapsed:.2f} | '
        f'ETA: 00:00:00'
    )

if __name__ == '__main__':
    asyncio.run(main())