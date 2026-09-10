import json

from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
from statistics import mean, median

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

def build_relationship_index(consolidated: list[dict]) -> dict[str, dict[str, dict]]:
    result = {}

    for item in consolidated:
        document_id = str(item['document_id'])
        relationship_id = str(item['relationship_id'])
        result.setdefault(document_id, {})[relationship_id] = item

    return result

def build_rst_index(rst_relations: list[dict]) -> dict[str, list[dict]]:
    result = {}

    for item in rst_relations:
        document_id = str(item['document_id'])
        result.setdefault(document_id, []).append(item)

    return result

def build_edu_relationship_index(relationships: dict[str, dict]) -> dict[int, set[str]]:
    result = {}

    for relationship_id, relationship in relationships.items():
        for edu_id in relationship['edu_ids']:
            edu_id = int(edu_id)
            result.setdefault(edu_id, set()).add(relationship_id)

    return result

def get_span_edus(sorted_edu_ids: list[int], start: int, end: int) -> list[int]:
    left = bisect_left(sorted_edu_ids, start)
    right = bisect_right(sorted_edu_ids, end)

    return sorted_edu_ids[left:right]

def build_tree_metrics(nodes: list[dict]) -> tuple[dict[str, int], dict[int, int]]:
    if not nodes:
        return {}, {}

    edu_count = max(node['right_end'] for node in nodes) + 1

    sorted_nodes = sorted(
        nodes,
        key=lambda node: (
            node['left_start'],
            -node['right_end'],
        ),
    )

    node_depths = {}
    stack = []

    for node in sorted_nodes:
        start = node['left_start']
        end = node['right_end']

        while stack:
            parent = stack[-1]

            if parent['left_start'] <= start and end <= parent['right_end']:
                break

            stack.pop()

        node_depths[node['rst_node_id']] = len(stack)
        stack.append(node)

    differences = [0] * (edu_count + 1)

    for node in nodes:
        start = node['left_start']
        end = node['right_end']

        differences[start] += 1

        if end + 1 < len(differences):
            differences[end + 1] -= 1

    edu_depths = {}
    depth = 0

    for edu_id in range(edu_count):
        depth += differences[edu_id]
        edu_depths[edu_id] = depth

    return node_depths, edu_depths

def build_edges(consolidated: list[dict], rst_relations: list[dict]) -> list[dict]:
    relationships_by_document = build_relationship_index(consolidated)
    rst_by_document = build_rst_index(rst_relations)
    edges = {}

    for document_id, relationships in relationships_by_document.items():
        nodes = rst_by_document.get(document_id, [])

        if not nodes:
            continue

        node_depths, edu_depths = build_tree_metrics(nodes)
        edu_relationships = build_edu_relationship_index(relationships)
        aligned_edu_ids = sorted(edu_relationships)

        for node in nodes:
            left_edu_ids = get_span_edus(aligned_edu_ids, node['left_start'], node['left_end'])
            right_edu_ids = get_span_edus(aligned_edu_ids, node['right_start'], node['right_end'])

            if not left_edu_ids or not right_edu_ids:
                continue

            lca_depth = node_depths[node['rst_node_id']]

            for left_edu_id in left_edu_ids:
                for right_edu_id in right_edu_ids:
                    left_hops = edu_depths[left_edu_id] - lca_depth
                    right_hops = edu_depths[right_edu_id] - lca_depth
                    tree_distance = left_hops + right_hops

                    for left_relationship_id in edu_relationships[left_edu_id]:
                        for right_relationship_id in edu_relationships[right_edu_id]:
                            if left_relationship_id == right_relationship_id:
                                continue

                            key = (
                                document_id,
                                node['rst_node_id'],
                                left_relationship_id,
                                right_relationship_id,
                            )

                            if key not in edges:
                                left_relationship = relationships[left_relationship_id]
                                right_relationship = relationships[right_relationship_id]

                                edges[key] = {
                                    'document_id': document_id,
                                    'rst_node_id': node['rst_node_id'],
                                    'relation': node['relation'],
                                    'nuclearity': node['nuclearity'],
                                    'left_role': node['left_role'],
                                    'right_role': node['right_role'],
                                    'left_relationship_id': left_relationship_id,
                                    'right_relationship_id': right_relationship_id,
                                    'left_source': left_relationship.get('source'),
                                    'left_target': left_relationship.get('target'),
                                    'right_source': right_relationship.get('source'),
                                    'right_target': right_relationship.get('target'),
                                    'lca_depth': lca_depth,
                                    'rst_span_start': node['left_start'],
                                    'rst_span_end': node['right_end'],
                                    'rst_span_size': node['right_end'] - node['left_start'] + 1,
                                    'left_rst_span': [node['left_start'], node['left_end']],
                                    'right_rst_span': [node['right_start'], node['right_end']],
                                    'left_support_edu_ids': set(),
                                    'right_support_edu_ids': set(),
                                    'support_pairs': {},
                                }

                            edge = edges[key]
                            edge['left_support_edu_ids'].add(left_edu_id)
                            edge['right_support_edu_ids'].add(right_edu_id)

                            pair_key = (left_edu_id, right_edu_id)

                            edge['support_pairs'][pair_key] = {
                                'left_edu_id': left_edu_id,
                                'right_edu_id': right_edu_id,
                                'left_hops': left_hops,
                                'right_hops': right_hops,
                                'tree_distance': tree_distance,
                            }

    result = []

    for edge in edges.values():
        edge['left_support_edu_ids'] = sorted(edge['left_support_edu_ids'])
        edge['right_support_edu_ids'] = sorted(edge['right_support_edu_ids'])
        edge['support_pairs'] = sorted(
            edge['support_pairs'].values(),
            key=lambda pair: (
                pair['left_edu_id'],
                pair['right_edu_id'],
            ),
        )

        distances = [
            pair['tree_distance']
            for pair in edge['support_pairs']
        ]

        edge['support_pair_count'] = len(edge['support_pairs'])
        edge['min_tree_distance'] = min(distances)
        edge['max_tree_distance'] = max(distances)
        edge['average_tree_distance'] = mean(distances)

        result.append(edge)

    result.sort(key=lambda item: (
        item['document_id'],
        item['rst_span_start'],
        item['rst_span_end'],
        item['left_relationship_id'],
        item['right_relationship_id'],
    ))

    return result

def percentile(values: list[int | float], p: float) -> int | float:
    if not values:
        return 0

    values = sorted(values)
    index = round((len(values) - 1) * p)

    return values[index]

def main():
    base_path = Path(__file__).resolve().parent
    consolidated_path = base_path / 'artifacts' / 'rgrag' / 'relationship_edu_consolidated.jsonl'
    rst_path = base_path / 'artifacts' / 'rgrag' / 'rst_relations.jsonl'
    output_path = base_path / 'artifacts' / 'rgrag' / 'rst_relationship_edges.jsonl'

    consolidated = load_jsonl(consolidated_path)
    rst_relations = load_jsonl(rst_path)

    edges = build_edges(consolidated, rst_relations)

    write_jsonl(output_path, edges)

    connected_relationships = set()
    documents = set()
    rst_nodes = set()
    relation_counts = Counter()
    nuclearity_counts = Counter()
    span_sizes = []
    min_distances = []
    support_pair_total = 0
    relationship_pair_counts = Counter()

    for edge in edges:
        left_relationship_id = edge['left_relationship_id']
        right_relationship_id = edge['right_relationship_id']

        connected_relationships.add(left_relationship_id)
        connected_relationships.add(right_relationship_id)
        documents.add(edge['document_id'])
        rst_nodes.add(edge['rst_node_id'])
        relation_counts[edge['relation']] += 1
        nuclearity_counts[edge['nuclearity']] += 1
        span_sizes.append(edge['rst_span_size'])
        min_distances.append(edge['min_tree_distance'])
        support_pair_total += edge['support_pair_count']

        pair_key = (
            edge['document_id'],
            *sorted([left_relationship_id, right_relationship_id]),
        )

        relationship_pair_counts[pair_key] += 1

    all_relationships = {
        str(item['relationship_id'])
        for item in consolidated
    }

    unique_relationship_pairs = len(relationship_pair_counts)

    multiple_rst_edges = sum(
        count > 1
        for count in relationship_pair_counts.values()
    )

    print('RST -> Relationship projection finished')
    print('Relationship-document groups:', len(consolidated))
    print('RST nodes:', len(rst_relations))
    print('RST nodes with projected edges:', len(rst_nodes))
    print('Projected RST edges:', len(edges))
    print('EDU support pairs:', support_pair_total)
    print('Unique relationship pairs:', unique_relationship_pairs)
    print('Relationship pairs with multiple RST edges:', multiple_rst_edges)
    print('Relationships connected:', len(connected_relationships))
    print('Relationships without RST edge:', len(all_relationships - connected_relationships))
    print('Documents with RST edges:', len(documents))

    if span_sizes:
        print(f'Median RST span size: {median(span_sizes):.0f}')
        print('P90 RST span size:', percentile(span_sizes, 0.90))
        print('Maximum RST span size:', max(span_sizes))

    if min_distances:
        print(f'Median minimum tree distance: {median(min_distances):.0f}')
        print('P90 minimum tree distance:', percentile(min_distances, 0.90))
        print('Maximum minimum tree distance:', max(min_distances))

    print('Nuclearities:')

    for nuclearity, count in nuclearity_counts.most_common():
        print(f'  {nuclearity}: {count}')

    print('Relations:')

    for relation, count in relation_counts.most_common():
        print(f'  {relation}: {count}')

    print('Output:', output_path)

if __name__ == '__main__':
    main()