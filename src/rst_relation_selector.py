from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

RELATION_DESCRIPTIONS = {
    'Joint': 'Two discourse units of equal importance are connected, coordinated, listed, or presented together.',
    'Background': 'One discourse unit provides contextual or background information needed to understand another.',
    'Elaboration': 'One discourse unit adds details, specifications, examples, or additional information about another.',
    'Temporal': 'The discourse units are related by time, temporal order, sequence, before, after, or simultaneous events.',
    'Topic-Change': 'The discourse shifts from one topic to another topic.',
    'Same-Unit': 'Separated discourse spans are parts of the same underlying discourse unit.',
    'Attribution': 'A statement, claim, or information is attributed to a speaker, source, author, or agent.',
    'TextualOrganization': 'The relation expresses how the text itself is organized, structured, or presented.',
    'Evaluation': 'One discourse unit evaluates, assesses, judges, or expresses an attitude toward another.',
    'Contrast': 'The discourse units express opposition, difference, contradiction, or contrasting properties.',
    'Explanation': 'One discourse unit explains why or how another statement, event, or situation holds.',
    'Enablement': 'One discourse unit enables, facilitates, or makes possible an action or outcome in another.',
    'Cause': 'One discourse unit describes a cause, reason, consequence, or causal relationship involving another.',
    'Topic-Comment': 'One discourse unit introduces a topic and another provides a comment or statement about that topic.',
    'Condition': 'One discourse unit expresses a condition, requirement, or circumstance under which another holds.',
    'Summary': 'One discourse unit summarizes, condenses, or restates the main information from another.',
    'Manner-Means': 'One discourse unit describes the manner, method, means, or way in which something is performed.',
    'Comparison': 'The discourse units are explicitly compared in terms of similarities, differences, properties, or outcomes.',
}

class RSTRelationSelection(BaseModel):
    relations: list[str] = Field(
        description='RST discourse relations relevant to the reasoning required by the question.'
    )

class RSTRelationSelector:
    def __init__(
        self,
        available_relations,
        base_url='http://127.0.0.1:1234/v1',
        api_key='lm-studio',
        model='google/gemma-4-e2b',
    ):
        self.available_relations = sorted(set(available_relations))

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0,
        )

        self.llm = llm.with_structured_output(RSTRelationSelection)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                'system',
                '''
You select RST discourse relations that may be useful for retrieving evidence required to answer a question.

Select only relations that describe useful discourse connections between pieces of evidence needed by the question.

Rules:
- Only select relations from the provided list.
- Do not select a relation merely because its name or description shares words with the question.
- Consider the reasoning and discourse connections required to answer the question.
- More than one relation may be relevant.
- Do not force unrelated relations.
'''.strip(),
            ),
            (
                'human',
                '''
Question:
{query}

Available RST relations:
{relations}
'''.strip(),
            ),
        ])

        self.chain = self.prompt | self.llm

    def build_relations_text(self) -> str:
        return '\n'.join(
            f'- {relation}: {RELATION_DESCRIPTIONS.get(relation, f"RST discourse relation of type {relation}.")}'
            for relation in self.available_relations
        )

    def select(self, query: str) -> list[str]:
        result = self.chain.invoke({
            'query': query,
            'relations': self.build_relations_text(),
        })

        return [
            relation
            for relation in result.relations
            if relation in self.available_relations
        ]