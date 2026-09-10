import json

from pathlib import Path

import lancedb
import numpy as np

from src.graphrag_retriever import GraphRAGRetriever

class TextUnitRanker:
    def __init__(self, graphrag: GraphRAGRetriever):
        self.graphrag = graphrag
        self.text_units = {
            str(text_unit.id): text_unit
            for text_unit in graphrag.text_units
        }

        config = graphrag.config.vector_store
        schema = config.index_schema['text_unit_text']

        database = lancedb.connect(config.db_uri)
        table = database.open_table(schema.index_name)
        dataframe = table.to_pandas()

        id_field = schema.id_field
        vector_field = schema.vector_field

        self.ids = [
            str(value)
            for value in dataframe[id_field]
        ]

        self.id_to_index = {
            text_unit_id: index
            for index, text_unit_id in enumerate(self.ids)
        }

        self.vectors = np.stack(
            dataframe[vector_field].to_numpy()
        ).astype(np.float32)

        norms = np.linalg.norm(
            self.vectors,
            axis=1,
            keepdims=True,
        )

        self.vectors = self.vectors / np.clip(
            norms,
            1e-12,
            None,
        )

    def get_query_vector(self, query: str) -> np.ndarray:
        text_embedder = (
            self.graphrag
            .search_engine
            .context_builder
            .text_embedder
        )

        vector = np.asarray(
            text_embedder.embedding(
                input=[query]
            ).first_embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(vector)

        return vector / max(norm, 1e-12)

    def rank(
        self,
        query: str,
        text_unit_ids: set[str],
        baseline_ids: set[str],
        rst_ids: set[str],
        top_k: int,
    ) -> list[dict]:
        query_vector = self.get_query_vector(query)
        result = []

        for text_unit_id in text_unit_ids:
            index = self.id_to_index.get(text_unit_id)

            if index is None:
                continue

            text_unit = self.text_units.get(text_unit_id)

            if text_unit is None:
                continue

            score = float(
                query_vector @ self.vectors[index]
            )

            if (
                text_unit_id in baseline_ids
                and text_unit_id in rst_ids
            ):
                origin = 'both'
            elif text_unit_id in baseline_ids:
                origin = 'graphrag'
            else:
                origin = 'rst'

            result.append({
                'text_unit_id': text_unit_id,
                'text': text_unit.text,
                'score': score,
                'origin': origin,
            })

        result.sort(
            key=lambda item: -item['score']
        )

        return result[:top_k]

class RGRAGRetriever:
    def __init__(
        self,
        workspace_path: str | Path,
        rst_edges_path: str | Path,
        consolidated_path: str | Path,
        community_level: int = 2,
    ):
        self.graphrag = GraphRAGRetriever(
            workspace_path=workspace_path,
            community_level=community_level,
        )

        self.rst_edges_path = Path(
            rst_edges_path
        )

        self.consolidated_path = Path(
            consolidated_path
        )

        self.rst_adjacency = (
            self.build_rst_adjacency()
        )

        self.relationship_context = (
            self.build_relationship_context()
        )

        self.ranker = TextUnitRanker(
            self.graphrag
        )

    def build_rst_adjacency(
        self,
    ) -> dict[str, list[dict]]:
        adjacency = {}

        with open(
            self.rst_edges_path,
            'r',
            encoding='utf-8',
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                edge = json.loads(line)

                left_id = str(
                    edge['left_relationship_id']
                )

                right_id = str(
                    edge['right_relationship_id']
                )

                adjacency.setdefault(
                    left_id,
                    [],
                ).append({
                    'relationship_id': right_id,
                    'document_id': str(
                        edge['document_id']
                    ),
                    'relation': edge['relation'],
                    'nuclearity': edge['nuclearity'],
                    'source_role': edge['left_role'],
                    'target_role': edge['right_role'],
                    'rst_node_id': edge['rst_node_id'],
                    'rst_span_size': edge['rst_span_size'],
                    'min_tree_distance': edge[
                        'min_tree_distance'
                    ],
                })

                adjacency.setdefault(
                    right_id,
                    [],
                ).append({
                    'relationship_id': left_id,
                    'document_id': str(
                        edge['document_id']
                    ),
                    'relation': edge['relation'],
                    'nuclearity': edge['nuclearity'],
                    'source_role': edge['right_role'],
                    'target_role': edge['left_role'],
                    'rst_node_id': edge['rst_node_id'],
                    'rst_span_size': edge['rst_span_size'],
                    'min_tree_distance': edge[
                        'min_tree_distance'
                    ],
                })

        return adjacency

    def build_relationship_context(
        self,
    ) -> dict[tuple[str, str], dict]:
        result = {}

        with open(
            self.consolidated_path,
            'r',
            encoding='utf-8',
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                key = (
                    str(item['relationship_id']),
                    str(item['document_id']),
                )

                result[key] = item

        return result

    def get_rst_candidates(
        self,
        relationship_ids: list[str],
    ) -> dict:
        seed_ids = {
            str(relationship_id)
            for relationship_id
            in relationship_ids
        }

        edges = []
        candidate_contexts = {}

        for seed_relationship_id in seed_ids:
            for edge in self.rst_adjacency.get(
                seed_relationship_id,
                [],
            ):
                relationship_id = edge[
                    'relationship_id'
                ]

                if relationship_id in seed_ids:
                    continue

                key = (
                    relationship_id,
                    edge['document_id'],
                )

                context = (
                    self.relationship_context.get(
                        key
                    )
                )

                if context is None:
                    continue

                edges.append({
                    'seed_relationship_id':
                        seed_relationship_id,
                    **edge,
                })

                candidate_contexts[key] = context

        relationship_ids = {
            relationship_id
            for relationship_id, _
            in candidate_contexts
        }

        text_unit_ids = set()

        for context in candidate_contexts.values():
            for text_unit_id in context.get(
                'text_unit_ids',
                [],
            ):
                text_unit_ids.add(
                    str(text_unit_id)
                )

        return {
            'edges': edges,
            'contexts': candidate_contexts,
            'relationship_ids': relationship_ids,
            'text_unit_ids': text_unit_ids,
        }

    def retrieve_candidates(
        self,
        query: str,
    ) -> dict:
        graphrag = (
            self.graphrag.retrieve_candidates(
                query
            )
        )

        rst = self.get_rst_candidates(
            graphrag['relationship_ids']
        )

        return {
            'graphrag': graphrag,
            'rst': rst,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> dict:
        candidates = self.retrieve_candidates(
            query
        )

        graphrag = candidates['graphrag']
        rst = candidates['rst']

        baseline_ids = {
            str(text_unit_id)
            for text_unit_id
            in graphrag['text_unit_ids']
        }

        rst_ids = {
            str(text_unit_id)
            for text_unit_id
            in rst['text_unit_ids']
        }

        rgrag_ids = (
            baseline_ids
            | rst_ids
        )

        baseline = self.ranker.rank(
            query=query,
            text_unit_ids=baseline_ids,
            baseline_ids=baseline_ids,
            rst_ids=set(),
            top_k=top_k,
        )

        rgrag = self.ranker.rank(
            query=query,
            text_unit_ids=rgrag_ids,
            baseline_ids=baseline_ids,
            rst_ids=rst_ids,
            top_k=top_k,
        )

        return {
            'graphrag': graphrag,
            'rst': rst,
            'baseline': baseline,
            'rgrag': rgrag,
            'baseline_candidate_count':
                len(baseline_ids),
            'rst_candidate_count':
                len(rst_ids),
            'rst_new_candidate_count':
                len(rst_ids - baseline_ids),
            'rgrag_candidate_count':
                len(rgrag_ids),
        }