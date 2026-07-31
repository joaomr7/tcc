from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

import numpy as np
from typing import Any
import json

from prompt import *

class Agents:
    def __init__(self, llm: ChatOpenAI, embed: OpenAIEmbeddings):
        self.llm = llm
        self.embed = embed
    
    def get_named_entities(self, text: str) -> list:
        # prepare prompt
        prompt = ChatPromptTemplate.from_template(get_named_entity_prompt())
        structured_llm = self.llm.with_structured_output(EntitiesResult)
        chain = prompt | structured_llm

        # call llm
        result = chain.invoke({'text' : text})
        return result.model_dump().get('entities', {})

    def get_context_vectors(self, context: list[str]):
        vectors = np.asarray(
            self.embed.embed_documents(context),
            dtype=np.float32
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / np.clip(norms, 1e-12, None)
        return normalized_vectors

    def get_top_k_context(self, query: str, vectors: np.ndarray, top_k: int = 5):
        query_vector = self.get_context_vectors([query])[0]
        similarities = query_vector @ vectors.T

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (idx, similarities[idx])
            for idx in top_indices
        ]

    def __get_similar_entities_keys(self, entities: dict, threshold: float = 0.60) -> list[tuple[str, str]]:
        if len(entities) < 2:
            return []
        
        keys = list(entities.keys())
        texts = [f'{entity["name"]} {entity["type"]}' for entity in entities.values()]

        # similarity
        vectors = self.get_context_vectors(texts)
        similarity_matrix = vectors @ vectors.T

        row_indices, column_indices = np.where(np.triu(similarity_matrix, k=1) > threshold)
        return [
            (
                keys[i],
                keys[j],
                float(similarity_matrix[i, j]),
            )
            for i, j in zip(row_indices, column_indices)
        ]

    def get_similar_entities_keys(self, entities: dict) -> list[tuple[str, str]]:
        # Initial candidates, with pre filter
        candidates = self.__get_similar_entities_keys(entities)

        # prepare prompt
        prompt = ChatPromptTemplate.from_template(get_entity_similarity_prompt())
        structured_llm = self.llm.with_structured_output(EntityComparisonResult)
        chain = prompt | structured_llm

        out_candidates = []
        for ents in candidates:
            ent1 = entities.get(ents[0])
            ent2 = entities.get(ents[1])

            # validate entity similarity with llm
            result = chain.invoke({
                'entity_1': json.dumps(ent1, ensure_ascii=False, sort_keys=True),
                'entity_2': json.dumps(ent2, ensure_ascii=False, sort_keys=True),
            }).result

            if result:
                out_candidates.append(ents)

        return out_candidates

    def extract_knowledge_graph(self, target_entity: str, ctx_text: str, related_kg: list[dict[str, Any]] | dict[str, Any] | None = None) -> dict[str, Any]:
        if not ctx_text or not ctx_text.strip():
            raise ValueError('Context text must not be empty')

        if not target_entity or not target_entity.strip():
            raise ValueError('Target entoty must not be empty')

        related_kg = related_kg or []
        related_kg_text = json.dumps(
            related_kg,
            ensure_ascii=False,
            indent=2,
            default=str
        )

        prompt = ChatPromptTemplate.from_template(get_knowledge_graph_prompt())
        structured_llm = self.llm.with_structured_output(KnowledgeGraphResult)
        chain = prompt | structured_llm

        result = chain.invoke({
            'text' : ctx_text,
            'target_entity' : target_entity,
            'related_kg' : related_kg_text
        })

        return result.model_dump()

class CustomRAKG:
    def __init__(self, llm: ChatOpenAI, embed: OpenAIEmbeddings):
        self.agents = Agents(llm, embed)

    def __identify_entities(self, entities: list, entity_num: int) -> dict:
        identified_entities = {}
        for offset, entity in enumerate(entities):
            identified_entities.update({
                f'entity{entity_num + offset}' : entity
            })
        
        return identified_entities

    def get_preliminary_entities(self, sents: list[str]) -> dict:
        all_entities = {}
        entity_num = 1

        for i, sent in enumerate(sents):
            # extract entities
            entities = self.agents.get_named_entities(sent)
            if not entities:
                continue

            # add sentence referene to entity
            for j in range(len(entities)):
                entities[j]['sent_id'] = i

            # create unique key for entities
            identified_entities = self.__identify_entities(entities, entity_num)
            entity_num += len(entities)
            
            all_entities.update(identified_entities)

        return all_entities
    
    def __merge_entities(self, entities: dict):
        similar = self.agents.get_similar_entities_keys(entities)

        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Initialize disjoint sets
        for ent in entities:
            parent[ent] = ent

        # Merge similar entities
        for a, b, _ in similar:
            if a in entities and b in entities:
                union(a, b)

        # Build groups
        groups = {}
        for ent in entities:
            root = find(ent)
            groups.setdefault(root, []).append(ent)

        merged_entities = {}

        # Merge each group
        for group in groups.values():
            main_ent = group[0]

            descriptions = []
            sent_ids = set()

            merged = entities[main_ent].copy()

            for ent in group:
                entity = entities[ent]

                # Merge descriptions
                desc = entity['description']
                if desc not in descriptions:
                    descriptions.append(desc)

                # Merge sentence ids
                sid = entity.get('sent_id')
                if isinstance(sid, list):
                    sent_ids.update(sid)
                elif sid is not None:
                    sent_ids.add(sid)

            merged['description'] = descriptions
            merged['sent_id'] = sorted(sent_ids)

            merged_entities[main_ent] = merged

        return merged_entities

    def format_indexed_sentences(
        self,
        sents: list[str],
        sentence_indices: set[int] | list[int]
    ) -> str:
        ordered_indices = sorted(sentence_indices)

        formatted_sentences = [
            f'Sentence {sent_idx}: {sents[sent_idx]}'
            for sent_idx in ordered_indices
        ]

        return '\n'.join(formatted_sentences)

    def get_kg_from_sents(self, sents: list[str]):
        # entity recognition
        preliminary_entities = self.get_preliminary_entities(sents)
        entities = self.__merge_entities(preliminary_entities)

        # build kg
        ctx_vectors = self.agents.get_context_vectors(sents)
        results = {}
        for id, ent in entities.items():
            sent_ids = set(ent.get('sent_id', []))
            ent_sents = [s for i, s in enumerate(sents) if i in sent_ids]
            if not ent_sents:
                print(f'Can\'t find sentences from entity: {ent}')
                continue

            name = ent.get('name', '')
            if not name:
                print(f'Can\'t find entity name from entity: {ent}')
                continue

            # retieve the top 5 context sentences
            top_ctx = self.agents.get_top_k_context(name, ctx_vectors, top_k=5)
            top_ctx_idx = set([int(i) for i, _ in top_ctx])

            # top context + entity sentences
            context_indices = sent_ids | top_ctx_idx
            ctx_text = self.format_indexed_sentences(
                sents=sents,
                sentence_indices=context_indices
            )

            result = self.agents.extract_knowledge_graph(name, ctx_text, None)
            results[id] = result

        return results

    def convert_kg(self, kg: dict):
        output = {
            'entities': [],
            'relations': []
        }

        entity_registry = {}
        relation_registry = {}

        # Register central entities
        for item in kg.values():
            central = item['central_entity']
            name = central['name']

            entity = entity_registry.setdefault(name, {
                'name': name,
                'type': central['type'],
                'description': central.get('description', ''),
                'description_source_sentences': [],
                'attributes': {}
            })

            # Merge description sources
            description_sources = central.get(
                'description_source_sentences',
                []
            )

            entity['description_source_sentences'] = sorted({
                *entity['description_source_sentences'],
                *description_sources
            })

            # Merge attributes
            for attr in central.get('attributes', []):
                key = attr['key']
                value = attr['value']
                source_sentences = attr.get('source_sentences', [])

                if key not in entity['attributes']:
                    entity['attributes'][key] = {
                        'value': value,
                        'source_sentences': sorted(set(source_sentences))
                    }
                else:
                    current_attribute = entity['attributes'][key]

                    current_attribute['source_sentences'] = sorted({
                        *current_attribute.get('source_sentences', []),
                        *source_sentences
                    })

        # Register target entities and relations
        for item in kg.values():
            central = item['central_entity']
            source = central['name']

            for rel in central.get('relationships', []):
                target = rel['target_name']

                target_entity = entity_registry.setdefault(target, {
                    'name': target,
                    'type': rel['target_type'],
                    'description': rel.get('target_description', ''),
                    'description_source_sentences': [],
                    'attributes': {}
                })

                # Keep a single entity type
                if not target_entity.get('type'):
                    target_entity['type'] = rel['target_type']

                relation_key = (
                    source,
                    rel['relation'],
                    target
                )

                source_sentences = rel.get('source_sentences', [])

                if relation_key not in relation_registry:
                    relation_registry[relation_key] = {
                        'source': source,
                        'relation': rel['relation'],
                        'target': target,
                        'description': rel.get(
                            'relation_description',
                            ''
                        ),
                        'source_sentences': sorted(
                            set(source_sentences)
                        )
                    }
                else:
                    existing_relation = relation_registry[relation_key]

                    existing_relation['source_sentences'] = sorted({
                        *existing_relation.get('source_sentences', []),
                        *source_sentences
                    })

        output['entities'] = sorted(
            entity_registry.values(),
            key=lambda x: x['name']
        )

        output['relations'] = sorted(
            relation_registry.values(),
            key=lambda x: (
                x['source'],
                x['relation'],
                x['target']
            )
        )

        return output
