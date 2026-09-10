from pathlib import Path

from src.rgrag_retriever import RGRAGRetriever
from src.rst_relation_selector import RSTRelationSelector

class RGRAGAnnotationRetriever:
    def __init__(
        self,
        workspace_path: str | Path,
        rst_edges_path: str | Path,
        consolidated_path: str | Path,
        community_level: int = 2,
        base_url: str = 'http://127.0.0.1:1234/v1',
        api_key: str = 'lm-studio',
        model: str = 'google/gemma-4-e2b',
    ):
        self.rgrag = RGRAGRetriever(
            workspace_path=workspace_path,
            rst_edges_path=rst_edges_path,
            consolidated_path=consolidated_path,
            community_level=community_level,
        )

        available_relations = {
            edge['relation']
            for edges in self.rgrag.rst_adjacency.values()
            for edge in edges
        }

        self.relation_selector = RSTRelationSelector(
            available_relations=available_relations,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )

    def get_annotation_candidates(
        self,
        relationship_ids: list[str],
        selected_relations: list[str],
    ) -> dict:
        seed_ids = {
            str(relationship_id)
            for relationship_id in relationship_ids
        }

        selected_relations = set(selected_relations)

        edges = []
        candidate_contexts = {}

        for seed_relationship_id in seed_ids:
            for edge in self.rgrag.rst_adjacency.get(seed_relationship_id, []):
                if edge['relation'] not in selected_relations:
                    continue

                relationship_id = str(edge['relationship_id'])

                if relationship_id in seed_ids:
                    continue

                key = (
                    relationship_id,
                    str(edge['document_id']),
                )

                context = self.rgrag.relationship_context.get(key)

                if context is None:
                    continue

                edges.append({
                    'seed_relationship_id': seed_relationship_id,
                    **edge,
                })

                candidate_contexts[key] = context

        candidate_relationship_ids = {
            relationship_id
            for relationship_id, _
            in candidate_contexts
        }

        text_unit_ids = set()

        for context in candidate_contexts.values():
            for text_unit_id in context.get('text_unit_ids', []):
                text_unit_ids.add(str(text_unit_id))

        return {
            'edges': edges,
            'contexts': candidate_contexts,
            'relationship_ids': candidate_relationship_ids,
            'text_unit_ids': text_unit_ids,
        }

    def retrieve_candidates(self, query: str) -> dict:
        graphrag = self.rgrag.graphrag.retrieve_candidates(query)

        selected_relations = self.relation_selector.select(query)

        annotation = self.get_annotation_candidates(
            relationship_ids=graphrag['relationship_ids'],
            selected_relations=selected_relations,
        )

        return {
            'graphrag': graphrag,
            'annotation': annotation,
            'selected_relations': selected_relations,
        }

    def retrieve(self, query: str, top_k: int = 10) -> dict:
        candidates = self.retrieve_candidates(query)

        graphrag = candidates['graphrag']
        annotation = candidates['annotation']

        graphrag_ids = {
            str(text_unit_id)
            for text_unit_id in graphrag['text_unit_ids']
        }

        annotation_ids = {
            str(text_unit_id)
            for text_unit_id in annotation['text_unit_ids']
        }

        rgrag_ids = graphrag_ids | annotation_ids

        ranked = self.rgrag.ranker.rank(
            query=query,
            text_unit_ids=rgrag_ids,
            baseline_ids=graphrag_ids,
            rst_ids=annotation_ids,
            top_k=top_k,
        )

        return {
            'candidates': candidates,
            'ranked': ranked,
            'selected_relations': candidates['selected_relations'],
            'graphrag_candidates': len(graphrag_ids),
            'annotation_candidates': len(annotation_ids),
            'annotation_new_candidates': len(annotation_ids - graphrag_ids),
            'rgrag_candidates': len(rgrag_ids),
            'annotation_relationships': len(annotation['relationship_ids']),
            'annotation_edges': len(annotation['edges']),
        }