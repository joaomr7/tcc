from pydantic import BaseModel, Field

# Entities
class Entity(BaseModel):
    name: str = Field(description='The main subject of the named entity')
    type: str = Field(description='The category of the subject')
    description: str = Field(description='A summary description explaining what the entity is')

class EntitiesResult(BaseModel):
    entities: list[Entity] = Field(description=('All named entities extracted from the text.'))

class EntityComparisonResult(BaseModel):
    result: bool

# KG
class KGAttribute(BaseModel):
    key: str = Field(description='Canonical name of the attribute.')
    value: str = Field(description='Literal value of the attribute.')
    source_sentences: list[int] = Field(
        default_factory=list,
        description=('Indices of the input sentences that explicitly support this attribute.')
    )

class KGRelationship(BaseModel):
    relation: str = Field(description='Short and canonical relationship name.')
    target_name: str = Field(description='Canonical name of the target entity.')
    target_type: str = Field(description='Canonical type of the target entity.')
    target_description: str = Field(description='Short description of the target entity.')
    relation_description: str = Field(description='Description of the relationship.')
    source_sentences: list[int] = Field(
        default_factory=list,
        description=('Indices of the input sentences that explicitly support this relationship.')
    )

class KGCentralEntity(BaseModel):
    name: str
    type: str
    description: str
    description_source_sentences: list[int] = Field(
        default_factory=list,
        description=('Indices of the input sentences that explicitly support the entity description.')
    )
    attributes: list[KGAttribute] = Field(default_factory=list)
    relationships: list[KGRelationship] = Field(default_factory=list)

class KnowledgeGraphResult(BaseModel):
    central_entity: KGCentralEntity

# Alignment
class ItemEduAlignment(BaseModel):
    item_id: str
    edu_ids: list[int]

class RSTEdgeSelection(BaseModel):
    source_item_id: str | None
    target_item_id: str | None

# KG Prompts
def get_named_entity_prompt() -> str:
    return '''
You are a named entity recognition assistant.

Text:
{text}

Task:
Extract all named entities mentioned in the text.

For each entity, provide:
- name: the canonical name of the entity.
- type: the most appropriate entity category.
- description: a concise description of the entity's role or significance in the provided text.

Guidelines:
1. Analyze the entire text before extracting entities.
2. Extract every relevant named entity explicitly mentioned in the text.
3. Include people, organizations, locations, products, software, datasets, methods, events, dates, and other named entities when appropriate.
4. The description should be specific to the context of the provided text, not a generic encyclopedia definition.
5. If an entity appears multiple times in the text, produce only one entry that summarizes all available information about that entity.
6. Do not extract common nouns, unnamed concepts, or generic references.
7. Do not infer information that is not supported by the text.
8. If the text contains no meaningful information or no named entities, return an empty entities list.
9. Return only the structured output defined by the schema.
'''.strip()

def get_entity_similarity_prompt() -> str:
    return '''
You are an entity resolution assistant.

Determine whether the following two JSON objects refer to the same real-world entity.

Entity 1:
{entity_1}

Entity 2:
{entity_2}

Rules:
1. Compare the canonical entity names.
2. Compare the entity types.
3. Use the descriptions only to resolve ambiguity.
4. Ignore differences in wording, formatting, capitalization, field order, and missing information.
5. If both entities have the same canonical name and compatible types, return True unless there is explicit evidence that they refer to different real-world entities.
6. Different descriptions do not imply different entities. They may describe different facts about the same entity.
7. Return False only when there is clear evidence that they represent different real-world entities.
'''.strip()

def get_knowledge_graph_prompt() -> str:
    return '''
You are a knowledge graph extraction assistant.

Text:
{text}

Target Entity:
{target_entity}

Related Knowledge Graphs:
{related_kg}

Task:
Extract a knowledge graph centered only on the target entity.

The input text is divided into numbered sentences. Each sentence is identified by its original document index using the format "Sentence N".

Rules:

1. Use only information explicitly supported by the provided text or the related knowledge graphs.

2. The output must describe only the target entity.

3. Every relationship must originate from the target entity.

4. Distinguish attributes from relationships:
   - Attributes are intrinsic properties of the target entity represented by literal values (text, numbers, dates, booleans, measurements, or other non-entity concepts).
   - Relationships connect the target entity to another identifiable named entity.
   - If the value of a fact is another named entity, it must always be represented as a relationship.
   - Attributes must never reference another named entity.
   - Never represent the same fact as both an attribute and a relationship.

5. Create an entity only if it:
   - is explicitly mentioned in the provided context;
   - has its own identity;
   - exists independently of the target entity;
   - can reasonably be represented as a standalone node in a knowledge graph.

6. Never create entities for actions, tasks, functionalities, capabilities, features, qualities, processes, descriptive phrases, or generic concepts, even if they appear as the object of a sentence. Represent such information as attributes or include them in descriptions.

7. Relationships must represent factual connections explicitly supported by the provided context. Do not infer new relationships.

8. Use canonical entity names and a single consistent type for each entity.

9. Entity descriptions must describe what the entity is, not the actions it performs or the relationships it participates in.

10. Use short, canonical relation names.

11. If the related knowledge graphs contain incoming relationships, generate the appropriate semantic inverse when necessary instead of copying the original predicate.

12. Remove duplicate attributes and relationships by merging semantically equivalent facts.

13. For every attribute, include the original indices of the sentence(s) that directly support that attribute in `source_sentences`.

14. For every relationship, include the original indices of the sentence(s) that directly support that relationship in `source_sentences`.

15. For the target entity description, include the original indices of the sentence(s) that directly support the description in `description_source_sentences`.

16. Use only sentence indices explicitly present in the provided text. Never invent, renumber, estimate, or infer sentence indices.

17. A sentence index may be assigned only if that sentence directly supports the exact fact being represented. A sentence that merely mentions the entity or provides contextual information must not be cited as evidence for an unrelated description, attribute, or relationship.

18. The sentence indices of an entity description must correspond only to sentences that explicitly describe or define that entity. Do not include sentences that describe only related entities or events.
'''.strip()

# KG RST Link Prompts

def get_alignment_prompt() -> str:
    return '''
You are aligning one knowledge graph item with Elementary Discourse Units (EDUs).

Select the EDU IDs that directly express the provided knowledge graph item.

Rules:
1. Use only the exact EDU IDs provided in Candidate EDUs.
2. Select only the EDU or EDUs that directly state the item.
3. Do not include EDUs that only provide unnecessary context.
4. Select the smallest possible set of EDUs.
5. Do not create, rewrite, rename, or correct the item.
6. For a relation, consider its source, relation, and target.
7. For an attribute, consider its entity, attribute name, and value.
8. If none of the candidate EDUs supports the item, return an empty list.
9. Preserve the item ID exactly as provided.
10. Do not return EDU IDs that are not explicitly provided.

Item:
{item}

Candidate EDUs:
{edus}
'''.strip()

def get_rst_edge_prompt() -> str:
    return '''
Select one knowledge graph item from the left candidates and one knowledge graph item from the right candidates that best represent the RST relation.

Rules:
1. Select only IDs from the provided candidates.
2. The source item must come from Left candidates.
3. The target item must come from Right candidates.
4. Prefer relations over attributes when both represent the same proposition.
5. Select the item that best represents the complete meaning of each text span.
6. Do not create, rewrite, rename, or correct item IDs.
7. Return null if no suitable item exists on one side.
8. Select only one item from each side.

RST relation:
{relation}

Left text:
{left_text}

Right text:
{right_text}

Left candidates:
{left_candidates}

Right candidates:
{right_candidates}
'''.strip()