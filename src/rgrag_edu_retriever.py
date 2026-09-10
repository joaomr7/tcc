import json

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.rgrag_retriever import RGRAGRetriever

class EDUTextUnitRanker:
    def __init__(
        self,
        rgrag: RGRAGRetriever,
        workspace_path: str | Path,
        rst_path: str | Path,
        embeddings_path: str | Path,
        index_path: str | Path,
    ):
        self.rgrag = rgrag
        self.edu_embeddings = np.load(embeddings_path, mmap_mode='r')

        with open(index_path, 'r', encoding='utf-8') as file:
            self.edu_index = json.load(file)

        self.row_to_edu_key = {
            row: edu_key
            for edu_key, row in self.edu_index.items()
        }

        self.edu_texts = {}
        self.edus_by_document = defaultdict(list)

        with open(rst_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)
                document_id = str(item['document_id'])

                for edu in item['edus']:
                    edu_key = edu['edu_key']

                    self.edu_texts[edu_key] = edu['text']

                    if edu_key not in self.edu_index:
                        continue

                    self.edus_by_document[document_id].append({
                        'edu_key': edu_key,
                        'row': self.edu_index[edu_key],
                        'start': int(edu['start']),
                        'end': int(edu['end']),
                    })

        documents_path = Path(workspace_path) / 'output' / 'documents.parquet'
        documents_df = pd.read_parquet(documents_path)

        self.document_texts = {
            str(row['id']): row['text']
            for _, row in documents_df.iterrows()
        }

        self.text_unit_edu_rows = self.build_text_unit_edu_rows()

    def build_text_unit_edu_rows(self) -> dict[str, list[int]]:
        mapping = {}

        for text_unit_id, text_unit in self.rgrag.ranker.text_units.items():
            document_id = str(text_unit.document_id)
            document_text = self.document_texts.get(document_id)

            if document_text is None:
                continue

            tu_start = document_text.find(text_unit.text)

            if tu_start < 0:
                continue

            tu_end = tu_start + len(text_unit.text)

            edu_rows = [
                edu['row']
                for edu in self.edus_by_document.get(document_id, [])
                if edu['start'] < tu_end and edu['end'] > tu_start
            ]

            if edu_rows:
                mapping[str(text_unit_id)] = edu_rows

        return mapping

    def get_origin(
        self,
        text_unit_id: str,
        graphrag_ids: set[str],
        rst_ids: set[str],
    ) -> str:
        if text_unit_id in graphrag_ids and text_unit_id in rst_ids:
            return 'both'

        if text_unit_id in graphrag_ids:
            return 'graphrag'

        return 'rst'

    def rank(
        self,
        query: str,
        text_unit_ids: set[str],
        graphrag_ids: set[str],
        rst_ids: set[str],
        top_k: int,
    ) -> list[dict]:
        query_vector = self.rgrag.ranker.get_query_vector(query)
        ranked = []

        for text_unit_id in text_unit_ids:
            text_unit_id = str(text_unit_id)

            text_unit = self.rgrag.ranker.text_units.get(text_unit_id)
            vector_index = self.rgrag.ranker.id_to_index.get(text_unit_id)

            if text_unit is None or vector_index is None:
                continue

            text_unit_score = float(
                query_vector @ self.rgrag.ranker.vectors[vector_index]
            )

            edu_rows = self.text_unit_edu_rows.get(text_unit_id, [])

            edu_score = None
            best_edu_key = None
            best_edu_text = None

            if edu_rows:
                vectors = np.asarray(
                    self.edu_embeddings[edu_rows],
                    dtype=np.float32,
                )

                scores = vectors @ query_vector

                best_index = int(np.argmax(scores))
                best_row = edu_rows[best_index]

                edu_score = float(scores[best_index])
                best_edu_key = self.row_to_edu_key[best_row]
                best_edu_text = self.edu_texts[best_edu_key]

            if edu_score is not None and edu_score > text_unit_score:
                score = edu_score
                score_source = 'edu'
            else:
                score = text_unit_score
                score_source = 'text_unit'

            ranked.append({
                'text_unit_id': text_unit_id,
                'text': text_unit.text,
                'score': score,
                'text_unit_score': text_unit_score,
                'edu_score': edu_score,
                'score_source': score_source,
                'best_edu_key': best_edu_key,
                'best_edu_text': best_edu_text,
                'origin': self.get_origin(
                    text_unit_id,
                    graphrag_ids,
                    rst_ids,
                ),
            })

        ranked.sort(key=lambda item: -item['score'])

        return ranked[:top_k]

class RGRAGEDURetriever:
    def __init__(
        self,
        workspace_path: str | Path,
        rst_edges_path: str | Path,
        consolidated_path: str | Path,
        rst_path: str | Path,
        embeddings_path: str | Path,
        index_path: str | Path,
        community_level: int = 2,
    ):
        self.rgrag = RGRAGRetriever(
            workspace_path=workspace_path,
            rst_edges_path=rst_edges_path,
            consolidated_path=consolidated_path,
            community_level=community_level,
        )

        self.ranker = EDUTextUnitRanker(
            rgrag=self.rgrag,
            workspace_path=workspace_path,
            rst_path=rst_path,
            embeddings_path=embeddings_path,
            index_path=index_path,
        )

    def retrieve(self, query: str, top_k: int = 10) -> dict:
        candidates = self.rgrag.retrieve_candidates(query)

        graphrag_result = candidates['graphrag']
        rst_result = candidates['rst']

        graphrag_ids = {
            str(text_unit_id)
            for text_unit_id in graphrag_result['text_unit_ids']
        }

        rst_ids = {
            str(text_unit_id)
            for text_unit_id in rst_result['text_unit_ids']
        }

        rgrag_ids = graphrag_ids | rst_ids

        ranked = self.ranker.rank(
            query=query,
            text_unit_ids=rgrag_ids,
            graphrag_ids=graphrag_ids,
            rst_ids=rst_ids,
            top_k=top_k,
        )

        return {
            'candidates': candidates,
            'ranked': ranked,
            'graphrag_candidates': len(graphrag_ids),
            'rst_new_candidates': len(rst_ids - graphrag_ids),
            'rgrag_candidates': len(rgrag_ids),
            'edu_covered_candidates': len(
                set(self.ranker.text_unit_edu_rows) & rgrag_ids
            ),
        }