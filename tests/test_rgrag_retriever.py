import sys

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import sys

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.rgrag_retriever import RGRAGRetriever

project_root = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

query = 'Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?'

retriever = RGRAGRetriever(
    workspace_path=(
        project_root
        / 'graphrag_workspace'
    ),
    rst_edges_path=(
        project_root
        / 'artifacts'
        / 'rgrag'
        / 'rst_relationship_edges.jsonl'
    ),
    consolidated_path=(
        project_root
        / 'artifacts'
        / 'rgrag'
        / 'relationship_edu_consolidated.jsonl'
    ),
    community_level=2,
)

result = retriever.retrieve(
    query=query,
    top_k=10,
)

baseline_ids = set(result['graphrag']['text_unit_ids'])
rst_ids = set(result['rst']['text_unit_ids'])
rst_only_ids = rst_ids - baseline_ids

rst_only = retriever.ranker.rank(
    query=query,
    text_unit_ids=rst_only_ids,
    baseline_ids=set(),
    rst_ids=rst_only_ids,
    top_k=len(rst_only_ids),
)

print()
print('RST ONLY')

for index, item in enumerate(rst_only, start=1):
    print()
    print(index, round(item['score'], 4), item['text_unit_id'])
    print(item['text'][:250])

print()
print('GRAPHRAG')
print(
    'Entities:',
    len(result['graphrag']['entities']),
)
print(
    'Relationships:',
    len(result['graphrag']['relationships']),
)
print(
    'TextUnits:',
    len(result['graphrag']['text_units']),
)

print()
print('RST EXPANSION')
print(
    'RST relationships:',
    len(result['rst']['relationship_ids']),
)
print(
    'RST TextUnits:',
    result['rst_candidate_count'],
)
print(
    'Novos RST TextUnits:',
    result['rst_new_candidate_count'],
)
print(
    'Total RGRAG candidates:',
    result['rgrag_candidate_count'],
)

print()
print('BASELINE')

for index, item in enumerate(
    result['baseline'],
    start=1,
):
    print()
    print(
        index,
        round(item['score'], 4),
        item['text_unit_id'],
    )
    print(
        item['text'][:250]
    )

print()
print('RGRAG')

for index, item in enumerate(
    result['rgrag'],
    start=1,
):
    print()
    print(
        index,
        item['origin'],
        round(item['score'], 4),
        item['text_unit_id'],
    )
    print(
        item['text'][:250]
    )