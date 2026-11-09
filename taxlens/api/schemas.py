"""Request and response models (docs/05-api-reference.md § 5.8)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from taxlens.agent.prompts import DISCLAIMER
from taxlens.retrieval.base import SearchFilters

DocTypeIn = Literal["act", "rules", "circular", "notification", "client_doc"]


class AskFilters(BaseModel):
    doc_types: list[DocTypeIn] | None = None
    act: str | None = None
    issued_after: date | None = None

    def to_search_filters(self) -> SearchFilters:
        return SearchFilters(doc_types=self.doc_types, act=self.act, issued_after=self.issued_after)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    filters: AskFilters | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class CitationOut(BaseModel):
    ref: str
    chunk_id: str
    title: str
    doc_type: str
    section: str | None = None
    page: int | None = None
    issued_on: date | None = None
    source_url: str | None = None


class SlabRowOut(BaseModel):
    lower: int
    upper: int
    rate: float  # 0.15 = 15%
    tax: int


class TaxResultOut(BaseModel):
    """The calculator's result, in whole rupees (amounts are already rounded)."""

    tax_year: str
    regime: Literal["new", "old"]
    gross_income: int
    standard_deduction: int
    deductions: int
    taxable_income: int
    tax_before_rebate: int
    rebate: int
    cess: int
    total_tax: int
    rules_source: str
    breakdown: list[SlabRowOut]


class AskResponse(BaseModel):
    answer: str
    route: Literal["qa", "calc", "refuse"]
    citations: list[CitationOut]
    found: bool
    disclaimer: str = DISCLAIMER
    trace_id: str | None = None
    prompt_version: str
    model: str
    calculation: TaxResultOut | None = None  # route "calc" only


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
