"""Typed models for a Redrob candidate profile.

Mirrors assets/schema/candidate_schema.json. Sections the scoring policy relies
on (career_history, redrob_signals, profile basics) are typed so malformed input
surfaces early; noisier sections (skills, certifications, languages) are kept
lenient because the dataset documents them as low-trust and partly synthetic.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# company_size and several proficiency fields are enums in the schema, but the
# dataset is known to contain noise, so they are typed as plain strings rather
# than Literals to avoid rejecting otherwise-usable profiles.


class Profile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: Optional[str] = None
    current_industry: str = ""


class CareerEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: str = ""
    title: str = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: Optional[str] = None
    description: str = ""


class Education(BaseModel):
    model_config = ConfigDict(extra="ignore")

    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    grade: Optional[str] = None
    tier: Optional[str] = None


class Skill(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    proficiency: Optional[str] = None
    endorsements: int = 0
    duration_months: int = 0


class Certification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    issuer: str = ""
    year: Optional[int] = None


class Language(BaseModel):
    model_config = ConfigDict(extra="ignore")

    language: str = ""
    proficiency: Optional[str] = None


class SalaryRange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min: float = 0.0
    max: float = 0.0


class RedrobSignals(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_completeness_score: float = 0.0
    signup_date: Optional[date] = None
    last_active_date: Optional[date] = None
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_range_inr_lpa: SalaryRange = Field(default_factory=SalaryRange)
    preferred_work_mode: Optional[str] = None
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    profile: Profile = Field(default_factory=Profile)
    career_history: list[CareerEntry] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)
