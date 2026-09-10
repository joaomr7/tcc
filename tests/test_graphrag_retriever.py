from pathlib import Path

from ..src.rgrag_retriever import RGRAGRetriever

project_root = Path(__file__).resolve().parent.parent

query = 'Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?'

retriever = RGRAGRetriever(
    workspace_path=project_root / 'graphrag_workspace',
    rst_edges_path=project_root / 'artifacts' / 'rgrag' / 'rst_relationship_edges.jsonl',
    community_level=2,
)

result = retriever.retrieve_candidates(query)

graphrag = result['graphrag']
rst = result['rst']

print()
print('GRAPHRAG')
print('Entities:', len(graphrag['entities']))
print('Relationships:', len(graphrag['relationships']))
print('TextUnits:', len(graphrag['text_units']))
print('In-network:', len(graphrag['in_network_relationships']))
print('Out-network:', len(graphrag['out_network_relationships']))

with_rst = [
    relationship_id
    for relationship_id in graphrag['relationship_ids']
    if relationship_id in retriever.rst_adjacency
]

print()
print('GRAPHRAG -> RST')
print('GraphRAG seeds:', len(graphrag['relationship_ids']))
print('Com conexão RST:', len(with_rst))
print('RST candidate edges:', len(rst['edges']))
print('RST candidate relationships:', len(rst['relationship_ids']))

print()
print('EXEMPLOS DE ARESTAS RST')

for index, edge in enumerate(rst['edges'][:10], start=1):
    print()
    print(
        index,
        edge['seed_relationship_id'],
        '->',
        edge['relationship_id'],
    )
    print(
        'Relation:',
        edge['relation'],
        '| Nuclearity:',
        edge['nuclearity'],
        '| Support pairs:',
        len(edge['support_pairs']),
    )

    if edge['support_pairs']:
        pair = edge['support_pairs'][0]

        print(
            'EDUs:',
            pair['source_edu_id'],
            '->',
            pair['target_edu_id'],
            '| Distance:',
            pair['tree_distance'],
        )

support_pair_count = 0
unique_support_pairs = set()

for edge in rst['edges']:
    support_pair_count += len(edge['support_pairs'])

    for pair in edge['support_pairs']:
        unique_support_pairs.add((
            edge['document_id'],
            pair['source_edu_id'],
            pair['target_edu_id'],
        ))

print()
print('RST SUPPORT PAIRS')
print('Total support pairs:', support_pair_count)
print('Unique support pairs:', len(unique_support_pairs))

