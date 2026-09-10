from pathlib import Path

import pandas as pd
from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.load_config import load_config
from graphrag.query.context_builder.entity_extraction import map_query_to_entities
from graphrag.query.context_builder.local_context import _filter_relationships
from graphrag.query.context_builder.source_context import count_relationships
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.retrieval.relationships import (
    get_in_network_relationships,
    get_out_network_relationships,
)
from graphrag.utils.api import get_embedding_store, load_search_prompt

class GraphRAGRetriever:
    def __init__(self, workspace_path: str | Path, community_level: int = 2):
        self.workspace_path = Path(workspace_path)
        self.output_path = self.workspace_path / 'output'
        self.community_level = community_level
        self.config = load_config(root_dir=self.workspace_path)

        self.entities_df = pd.read_parquet(self.output_path / 'entities.parquet')
        self.communities_df = pd.read_parquet(self.output_path / 'communities.parquet')
        self.reports_df = pd.read_parquet(self.output_path / 'community_reports.parquet')
        self.relationships_df = pd.read_parquet(self.output_path / 'relationships.parquet')
        self.text_units_df = pd.read_parquet(self.output_path / 'text_units.parquet')

        covariates_path = self.output_path / 'covariates.parquet'
        self.covariates_df = pd.read_parquet(covariates_path) if covariates_path.exists() else None

        self.entities = read_indexer_entities(
            self.entities_df,
            self.communities_df,
            self.community_level,
        )

        self.reports = read_indexer_reports(
            self.reports_df,
            self.communities_df,
            self.community_level,
        )

        self.relationships = read_indexer_relationships(self.relationships_df)
        self.text_units = read_indexer_text_units(self.text_units_df)
        self.covariates = read_indexer_covariates(self.covariates_df) if self.covariates_df is not None else []

        self.search_engine = self.build_search_engine()

    def build_search_engine(self):
        description_embedding_store = get_embedding_store(
            config=self.config.vector_store,
            embedding_name=entity_description_embedding,
        )

        prompt = load_search_prompt(self.config.local_search.prompt)

        return get_local_search_engine(
            config=self.config,
            reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates={'claims': self.covariates},
            response_type='Multiple Paragraphs',
            description_embedding_store=description_embedding_store,
            system_prompt=prompt,
        )

    def get_selected_entities(self, query: str) -> list:
        context_builder = self.search_engine.context_builder
        params = self.search_engine.context_builder_params

        return map_query_to_entities(
            query=query,
            text_embedding_vectorstore=context_builder.entity_text_embeddings,
            text_embedder=context_builder.text_embedder,
            all_entities_dict=context_builder.entities,
            embedding_vectorstore_key=context_builder.embedding_vectorstore_key,
            include_entity_names=[],
            exclude_entity_names=[],
            k=params['top_k_mapped_entities'],
            oversample_scaler=2,
        )

    def get_prioritized_relationships(self, selected_entities: list) -> list:
        context_builder = self.search_engine.context_builder
        params = self.search_engine.context_builder_params

        return _filter_relationships(
            selected_entities=selected_entities,
            relationships=list(context_builder.relationships.values()),
            top_k_relationships=params['top_k_relationships'],
            relationship_ranking_attribute=params.get(
                'relationship_ranking_attribute',
                'rank',
            ),
        )

    def get_prioritized_text_units(self, selected_entities: list) -> list:
        context_builder = self.search_engine.context_builder
        relationships = list(context_builder.relationships.values())

        unit_info = []
        seen_ids = set()

        for entity_index, entity in enumerate(selected_entities):
            entity_relationships = [
                relationship
                for relationship in relationships
                if relationship.source == entity.title
                or relationship.target == entity.title
            ]

            for text_unit_id in entity.text_unit_ids or []:
                if text_unit_id in seen_ids:
                    continue

                text_unit = context_builder.text_units.get(text_unit_id)

                if text_unit is None:
                    continue

                relationship_count = count_relationships(
                    entity_relationships,
                    text_unit,
                )

                unit_info.append(
                    (
                        text_unit,
                        entity_index,
                        relationship_count,
                    )
                )

                seen_ids.add(text_unit_id)

        unit_info.sort(
            key=lambda item: (
                item[1],
                -item[2],
            )
        )

        return [
            item[0]
            for item in unit_info
        ]

    def retrieve_candidates(self, query: str) -> dict:
        selected_entities = self.get_selected_entities(query)

        prioritized_relationships = self.get_prioritized_relationships(
            selected_entities
        )

        prioritized_text_units = self.get_prioritized_text_units(
            selected_entities
        )

        all_relationships = list(
            self.search_engine.context_builder.relationships.values()
        )

        in_network = get_in_network_relationships(
            selected_entities=selected_entities,
            relationships=all_relationships,
            ranking_attribute='rank',
        )

        out_network = get_out_network_relationships(
            selected_entities=selected_entities,
            relationships=all_relationships,
            ranking_attribute='rank',
        )

        return {
            'entities': selected_entities,
            'relationships': prioritized_relationships,
            'text_units': prioritized_text_units,
            'in_network_relationships': in_network,
            'out_network_relationships': out_network,
            'entity_ids': [
                str(entity.id)
                for entity in selected_entities
            ],
            'relationship_ids': [
                str(relationship.id)
                for relationship in prioritized_relationships
            ],
            'text_unit_ids': [
                str(text_unit.id)
                for text_unit in prioritized_text_units
            ],
        }