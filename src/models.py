from pydantic import BaseModel, Field

class Document(BaseModel):
    document_id: str
    text: str

class KGEdge(BaseModel):
    source: str
    relation: str
    target: str
    document_id: str
    source_text: str | None = None
    char_start: int | None = None
    char_end: int | None = None

class EDU(BaseModel):
    edu_id: str
    document_id: str
    text: str
    char_start: int
    char_end: int

class RSTRelation(BaseModel):
    source_edu: str
    target_edu: str
    relation: str
    source_nuclearity: str | None = None
    target_nuclearity: str | None = None

class EnrichedKGEdge(KGEdge):
    source_edu: str | None = None
    rst_relations: list[RSTRelation] = Field(default_factory=list)
    alignment_status: str = 'not_processed'