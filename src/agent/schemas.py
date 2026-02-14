from __future__ import annotations

from datetime import date
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator

Department = Literal[
    "Artificial Intelligence",
    "AI/ML",
    "Machine Learning",
    "Data Science",
]


class Employee(BaseModel):
    user_id: int
    name: str = Field(min_length=1)
    age: Optional[int] = Field(default=None, ge=16, le=80)
    email: Optional[str] = None
    salary: Optional[float] = Field(default=None, ge=0)
    join_date: Optional[date] = None
    department: Department
    performance_score: Optional[float] = Field(default=None, ge=0, le=10)
    location: Optional[str] = None
    job_title: Optional[str] = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        v = " ".join(str(v).strip().split())
        return v.title()


class RejectedRecord(BaseModel):
    raw_record: str = Field(min_length=1)
    reasons: List[str] = Field(min_length=1)


class ExtractedData(BaseModel):
    # Valid, schema-compliant records only
    employees: List[Employee] = Field(default_factory=list)

    # Records that cannot be made valid WITHOUT guessing required fields
    rejected: List[RejectedRecord] = Field(default_factory=list)
