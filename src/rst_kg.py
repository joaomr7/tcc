from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from typing import Dict, List, Union
import json
import re

from prompt import ItemEduAlignment, RSTEdgeSelection, get_alignment_prompt, get_rst_edge_prompt
from kg import CustomRAKG
from rst import DMRSTParser

class RKG:
    _RST_NODE_PATTERN = re.compile(
        r'''
        \(
            (?P<left_start>\d+):
            (?P<left_role>Nucleus|Satellite)=
            (?P<left_relation>[^:,\s]+):
            (?P<left_end>\d+),
            (?P<right_start>\d+):
            (?P<right_role>Nucleus|Satellite)=
            (?P<right_relation>[^:,\s]+):
            (?P<right_end>\d+)
        \)
        ''',
        re.VERBOSE,
    )

    def __init__(
        self,
        llm: ChatOpenAI,
        embed: OpenAIEmbeddings,
        dmrst_checkpoint_path: Union[str, Path],
        dmrst_cache_dir: Union[str, Path],
        dmrst_batch_size: int = 1,
    ):
        self.llm = llm
        self.crakg = CustomRAKG(llm, embed)
        self.dmrst = DMRSTParser(dmrst_checkpoint_path, dmrst_cache_dir, dmrst_batch_size)

    def __normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)

        return text.strip()

    def __remove_punctuation(self, text: str) -> str:
        return re.sub(
            r'[,.!?;:]',
            '',
            self.__normalize_text(text),
        )

    def __align_sentences_to_edus(self, sentences: list[str], edus: list[str]) -> dict[int, list[int]]:
        sentence_to_edus = {}
        edu_cursor = 0

        for sentence_id, sentence in enumerate(sentences):
            target = self.__remove_punctuation(sentence)

            accumulated = ''
            matched_edu_ids = []

            while edu_cursor < len(edus):
                edu_id = edu_cursor

                accumulated = self.__normalize_text(f'{accumulated} {edus[edu_cursor]}')

                matched_edu_ids.append(edu_id)
                edu_cursor += 1

                current = self.__remove_punctuation(accumulated)

                if current == target:
                    break

                if not target.startswith(current):
                    raise ValueError(f'Could not align sentence {sentence_id} with the EDU sequence. Current text: {current!r}')

            if self.__remove_punctuation(accumulated) != target:
                raise ValueError(f'Incomplete EDU alignment for sentence {sentence_id}.')

            sentence_to_edus[sentence_id] = matched_edu_ids

        if edu_cursor != len(edus):
            raise ValueError('Some EDUs were not aligned with any sentence.')

        return sentence_to_edus

    def __extract_kg_items(self, kg: dict) -> list[dict]:
        items = []

        for index, relation in enumerate(
            kg.get('relations', [])
        ):
            items.append({
                'id': f'relation_{index}',
                'type': 'relation',
                'source': relation.get('source'),
                'relation': relation.get('relation'),
                'target': relation.get('target'),
                'source_sentences': relation.get(
                    'source_sentences',
                    [],
                ),
            })

        for entity_index, entity in enumerate(
            kg.get('entities', [])
        ):
            for attribute_name, attribute in entity.get(
                'attributes',
                {},
            ).items():
                items.append({
                    'id': (
                        f'entity_{entity_index}_'
                        f'attribute_{attribute_name}'
                    ),
                    'type': 'attribute',
                    'entity': entity.get('name'),
                    'attribute': attribute_name,
                    'value': attribute.get('value'),
                    'source_sentences': attribute.get(
                        'source_sentences',
                        [],
                    ),
                })

        return items

    def __get_candidate_edus(self, item: dict, sentence_to_edus: dict[int, list[int]], edus: list[str]) -> list[dict]:
        candidate_ids = sorted({
            edu_id
            for sentence_id in item.get('source_sentences', [])
            for edu_id in sentence_to_edus.get(sentence_id, [])
        })

        return [
            {
                'edu_id': edu_id,
                'text': edus[edu_id],
            }
            for edu_id in candidate_ids
        ]

    def __align_items_to_edus(self, items: list[dict], edus: list[str], sentence_to_edus: dict[int, list[int]]) -> dict[str, list[int]]:
        prompt = ChatPromptTemplate.from_template(get_alignment_prompt())
        structured_llm = self.llm.with_structured_output(ItemEduAlignment)

        chain = prompt | structured_llm

        result = {}

        for item in items:
            candidate_edus = self.__get_candidate_edus(
                item=item,
                sentence_to_edus=sentence_to_edus,
                edus=edus,
            )

            if not candidate_edus:
                result[item['id']] = []
                continue

            response = chain.invoke({
                'item': json.dumps(
                    item,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                'edus': json.dumps(
                    candidate_edus,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            })

            if response.item_id != item['id']:
                raise ValueError(f'Invalid item ID returned by the LLM. Expected {item["id"]!r}, received {response.item_id!r}.')

            valid_edu_ids = {edu['edu_id'] for edu in candidate_edus}
            invalid_edu_ids = (set(response.edu_ids) - valid_edu_ids)

            if invalid_edu_ids:
                raise ValueError(f'Invalid EDU IDs returned for {item["id"]}: {sorted(invalid_edu_ids)}. Valid IDs: {sorted(valid_edu_ids)}')

            result[item['id']] = sorted(
                set(response.edu_ids)
            )

        return result

    def __parse_rst_tree(self, tree: str) -> list[dict]:
        nodes = []

        for index, match in enumerate(
            self._RST_NODE_PATTERN.finditer(tree)
        ):
            data = match.groupdict()

            left_role = data['left_role']
            right_role = data['right_role']

            nuclearity = (
                ('N' if left_role == 'Nucleus' else 'S')
                + ('N' if right_role == 'Nucleus' else 'S')
            )

            left_relation = data['left_relation']
            right_relation = data['right_relation']

            if left_relation != 'span':
                relation = left_relation

            elif right_relation != 'span':
                relation = right_relation

            else:
                relation = 'span'

            original_left_span = [int(data['left_start']), int(data['left_end'])]

            original_right_span = [int(data['right_start']), int(data['right_end'])]

            nodes.append({
                'id': f'rst_{index}',
                'left_span': [
                    original_left_span[0] - 1,
                    original_left_span[1] - 1,
                ],
                'right_span': [
                    original_right_span[0] - 1,
                    original_right_span[1] - 1,
                ],
                'original_left_span': original_left_span,
                'original_right_span': original_right_span,
                'relation': relation,
                'nuclearity': nuclearity,
                'left_role': left_role,
                'right_role': right_role,
            })

        return nodes

    def __get_items_in_span(self, item_edus: dict[str, list[int]], span: list[int]) -> list[str]:
        start, end = span

        return [
            item_id
            for item_id, edu_ids in item_edus.items()
            if any(
                start <= edu_id <= end
                for edu_id in edu_ids
            )
        ]

    def __get_edus_in_span(self, edus: list[str], span: list[int]) -> list[dict]:
        start, end = span

        return [
            {
                'edu_id': edu_id,
                'text': edus[edu_id],
            }
            for edu_id in range(start, end + 1)
        ]

    def __join_span_text(self, edus: list[str], span: list[int]) -> str:
        start, end = span

        return ' '.join(
            edus[edu_id]
            for edu_id in range(start, end + 1)
        )

    def __select_rst_edge_items(
        self,
        node: dict,
        left_item_ids: list[str],
        right_item_ids: list[str],
        items_by_id: dict[str, dict],
        edus: list[str],
    ) -> tuple[str | None, str | None]:
        if not left_item_ids or not right_item_ids:
            return None, None

        if (
            len(left_item_ids) == 1
            and len(right_item_ids) == 1
        ):
            return left_item_ids[0], right_item_ids[0]

        prompt = ChatPromptTemplate.from_template(get_rst_edge_prompt())
        structured_llm = self.llm.with_structured_output(RSTEdgeSelection)

        chain = prompt | structured_llm

        left_candidates = [
            items_by_id[item_id]
            for item_id in left_item_ids
        ]

        right_candidates = [
            items_by_id[item_id]
            for item_id in right_item_ids
        ]

        response = chain.invoke({
            'relation': node['relation'],
            'left_text': self.__join_span_text(
                edus=edus,
                span=node['left_span'],
            ),
            'right_text': self.__join_span_text(
                edus=edus,
                span=node['right_span'],
            ),
            'left_candidates': json.dumps(
                left_candidates,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            'right_candidates': json.dumps(
                right_candidates,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        })

        if (response.source_item_id is not None and response.source_item_id not in left_item_ids):
            raise ValueError(f'Invalid source item returned by the LLM: {response.source_item_id}')

        if (response.target_item_id is not None and response.target_item_id not in right_item_ids):
            raise ValueError(f'Invalid target item returned by the LLM: {response.target_item_id}')

        return (response.source_item_id, response.target_item_id)

    def __build_rst_edges(
        self,
        rst_nodes: list[dict],
        item_edus: dict[str, list[int]],
        items: list[dict],
        edus: list[str],
    ) -> list[dict]:
        items_by_id = {item['id']: item for item in items}

        edges = []

        for node in rst_nodes:
            left_item_ids = self.__get_items_in_span(
                item_edus=item_edus,
                span=node['left_span'],
            )

            right_item_ids = self.__get_items_in_span(
                item_edus=item_edus,
                span=node['right_span'],
            )

            source_item_id, target_item_id = (
                self.__select_rst_edge_items(
                    node=node,
                    left_item_ids=left_item_ids,
                    right_item_ids=right_item_ids,
                    items_by_id=items_by_id,
                    edus=edus,
                )
            )

            if source_item_id is None:
                continue

            if target_item_id is None:
                continue

            if source_item_id == target_item_id:
                continue

            edges.append({
                'source_item': source_item_id,
                'target_item': target_item_id,
                'relation': node['relation'],
                'nuclearity': node['nuclearity'],
                'source_role': node['left_role'],
                'target_role': node['right_role'],
                'source_span': node['left_span'],
                'target_span': node['right_span'],
                'rst_node_id': node['id'],
            })

        return edges

    def __enrich_kg_with_rst(self, kg: dict, rst_output: dict, sentences: list[str]) -> dict:
        edus = rst_output['edus']
        tree = rst_output['tree']

        sentence_to_edus = self.__align_sentences_to_edus(
            sentences=sentences,
            edus=edus,
        )

        items = self.__extract_kg_items(kg)

        item_edus = self.__align_items_to_edus(
            items=items,
            edus=edus,
            sentence_to_edus=sentence_to_edus,
        )

        rst_nodes = self.__parse_rst_tree(tree)

        rst_edges = self.__build_rst_edges(
            rst_nodes=rst_nodes,
            item_edus=item_edus,
            items=items,
            edus=edus,
        )

        return {
            'sentence_to_edus': sentence_to_edus,
            'items': items,
            'item_edus': item_edus,
            'rst_nodes': rst_nodes,
            'rst_edges': rst_edges,
        }

    def get_rkg_from_sentences(self, sents: list[str]):
        temp_kg = self.crakg.get_kg_from_sents(sents)
        kg = self.crakg.convert_kg(temp_kg)
        rst = self.dmrst.parse(' '.join(sents))
        rst_enrichment = self.__enrich_kg_with_rst(kg, rst, sents)

        return {
            'knowledge_graph' : kg,
            'rst' : {
                'edus' : rst['edus'],
                'tree' : rst['tree']
            },
            'rst_enrichment' : rst_enrichment
        }

    def get_rkg_from_document(self, document: str):
        pattern = re.compile(r'(?<!\b[A-Za-z]\.)(?<=[.!?。！？])\s+')
        sentences = [s.strip() for s in pattern.split(document) if s.strip()]

        return {
            'document' : document,
            'sentences' : sentences,
            'rstkg' : self.get_rkg_from_sentences(sentences)
        }