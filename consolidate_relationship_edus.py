import json

from pathlib import Path

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

def get_group_key(item: dict) -> tuple[str, str]:
    return (
        str(item['relationship_id']),
        str(item['document_id']),
    )

def consolidate_alignments(alignments: list[dict]) -> list[dict]:
    groups = {}

    for item in alignments:
        key = get_group_key(item)

        if key not in groups:
            groups[key] = {
                'relationship_id': str(item['relationship_id']),
                'source': item.get('source'),
                'target': item.get('target'),
                'description': item.get('description'),
                'document_id': str(item['document_id']),
                'context_count': 0,
                'non_empty_context_count': 0,
                'text_unit_ids': [],
                'text_unit_id_set': set(),
                'edus': {},
            }

        group = groups[key]
        group['context_count'] += 1

        text_unit_id = str(item['text_unit_id'])

        if text_unit_id not in group['text_unit_id_set']:
            group['text_unit_id_set'].add(text_unit_id)
            group['text_unit_ids'].append(text_unit_id)

        if item.get('selected_edu_count', 0) > 0:
            group['non_empty_context_count'] += 1

        for edu in item.get('selected_edus', []):
            edu_id = int(edu['edu_id'])

            if edu_id not in group['edus']:
                group['edus'][edu_id] = {
                    'edu_id': edu_id,
                    'edu_key': edu['edu_key'],
                    'text': edu['text'],
                }

    result = []

    for key in sorted(groups):
        group = groups[key]
        edus = [group['edus'][edu_id] for edu_id in sorted(group['edus'])]

        result.append({
            'relationship_id': group['relationship_id'],
            'source': group['source'],
            'target': group['target'],
            'description': group['description'],
            'document_id': group['document_id'],
            'context_count': group['context_count'],
            'non_empty_context_count': group['non_empty_context_count'],
            'text_unit_count': len(group['text_unit_ids']),
            'text_unit_ids': group['text_unit_ids'],
            'edu_count': len(edus),
            'edu_ids': [edu['edu_id'] for edu in edus],
            'edu_keys': [edu['edu_key'] for edu in edus],
            'edus': edus,
        })

    return result

def main():
    base_path = Path(__file__).resolve().parent
    alignment_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_alignment.jsonl'
    output_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_consolidated.jsonl'

    alignments = load_jsonl(alignment_path)
    consolidated = consolidate_alignments(alignments)

    write_jsonl(output_path, consolidated)

    relationships = {item['relationship_id'] for item in consolidated}
    documents = {item['document_id'] for item in consolidated}
    groups_with_edus = sum(item['edu_count'] > 0 for item in consolidated)
    groups_without_edus = sum(item['edu_count'] == 0 for item in consolidated)
    total_edu_links = sum(item['edu_count'] for item in consolidated)
    average_edus = total_edu_links / groups_with_edus if groups_with_edus else 0

    relationship_documents = {}

    for item in consolidated:
        relationship_documents.setdefault(item['relationship_id'], set()).add(item['document_id'])

    multi_document_relationships = sum(
        len(document_ids) > 1
        for document_ids in relationship_documents.values()
    )

    print('Relationship -> EDU consolidation finished')
    print('Input aligned contexts:', len(alignments))
    print('Relationships:', len(relationships))
    print('Documents:', len(documents))
    print('Relationship-document groups:', len(consolidated))
    print('Groups with EDUs:', groups_with_edus)
    print('Groups without EDUs:', groups_without_edus)
    print('Relationships in multiple documents:', multi_document_relationships)
    print('Unique Relationship-EDU links:', total_edu_links)
    print(f'Average EDUs per non-empty group: {average_edus:.2f}')
    print('Output:', output_path)

if __name__ == '__main__':
    main()