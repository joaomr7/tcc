import json
import re

from collections import Counter
from pathlib import Path

RST_NODE_PATTERN = re.compile(
    r'\((\d+):(Nucleus|Satellite)=([^:,\s]+):(\d+),'
    r'(\d+):(Nucleus|Satellite)=([^:,\s]+):(\d+)\)'
)

def load_jsonl(path: str | Path) -> list[dict]:
    result = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line:
                result.append(json.loads(line))

    return result

def write_jsonl(path: str | Path, data: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_nuclearity(left_role: str, right_role: str) -> str:
    left = 'N' if left_role == 'Nucleus' else 'S'
    right = 'N' if right_role == 'Nucleus' else 'S'
    return left + right

def get_relation(nuclearity: str, left_relation: str, right_relation: str) -> str:
    if nuclearity == 'NN':
        if left_relation != right_relation:
            raise ValueError(f'NN node with different relations: {left_relation}, {right_relation}')

        return left_relation

    if nuclearity == 'NS':
        if left_relation != 'span':
            raise ValueError(f'NS node with invalid nucleus relation: {left_relation}')

        return right_relation

    if nuclearity == 'SN':
        if right_relation != 'span':
            raise ValueError(f'SN node with invalid nucleus relation: {right_relation}')

        return left_relation

    raise ValueError(f'Unsupported nuclearity: {nuclearity}')

def parse_document(document: dict) -> list[dict]:
    document_id = str(document['document_id'])
    tree = document['tree']
    edu_count = int(document['edu_count'])
    result = []

    for node_index, match in enumerate(RST_NODE_PATTERN.finditer(tree)):
        left_start = int(match.group(1)) - 1
        left_role = match.group(2)
        left_relation = match.group(3)
        left_end = int(match.group(4)) - 1
        right_start = int(match.group(5)) - 1
        right_role = match.group(6)
        right_relation = match.group(7)
        right_end = int(match.group(8)) - 1

        nuclearity = get_nuclearity(left_role, right_role)
        relation = get_relation(nuclearity, left_relation, right_relation)

        if not 0 <= left_start <= left_end < edu_count:
            raise ValueError(f'Invalid left span in document {document_id}: {left_start}-{left_end}')

        if not 0 <= right_start <= right_end < edu_count:
            raise ValueError(f'Invalid right span in document {document_id}: {right_start}-{right_end}')

        if left_end >= right_start:
            raise ValueError(f'Overlapping RST spans in document {document_id}: {left_start}-{left_end}, {right_start}-{right_end}')

        result.append({
            'rst_node_id': f'{document_id}:{node_index}',
            'document_id': document_id,
            'relation': relation,
            'nuclearity': nuclearity,
            'left_role': left_role,
            'left_start': left_start,
            'left_end': left_end,
            'right_role': right_role,
            'right_start': right_start,
            'right_end': right_end,
        })

    return result

def main():
    base_path = Path(__file__).resolve().parent
    rst_path = base_path / 'artifacts' / 'rst' / 'multihop_rst.jsonl'
    output_path = base_path / 'artifacts' / 'rgrag' / 'rst_relations.jsonl'

    documents = load_jsonl(rst_path)
    relations = []

    for document in documents:
        relations.extend(parse_document(document))

    write_jsonl(output_path, relations)

    nuclearities = Counter(item['nuclearity'] for item in relations)
    relation_counts = Counter(item['relation'] for item in relations)

    print('RST parsing finished')
    print('Documents:', len(documents))
    print('RST nodes:', len(relations))
    print('Nuclearities:')

    for nuclearity, count in nuclearities.most_common():
        print(f'  {nuclearity}: {count}')

    print('Relations:')

    for relation, count in relation_counts.most_common():
        print(f'  {relation}: {count}')

    print('Output:', output_path)

if __name__ == '__main__':
    main()