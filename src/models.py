"""Data models for the Hiring Feasibility Engine."""

from dataclasses import dataclass
from datetime import date
from src.config import TODAY, COMPLEXITY_SHAPE_K


@dataclass
class Recruiter:
    id: str
    name: str
    avg_monthly_capacity: float
    
    @property
    def daily_rate(self) -> float:
        return self.avg_monthly_capacity / 30.44
    
    @property
    def mean_days_between_hires(self) -> float:
        return 30.44 / self.avg_monthly_capacity


@dataclass
class Role:
    id: str
    role_name: str
    complexity: str
    avg_days_to_hire: int
    target_start_date: date
    assigned_recruiter_id: str
    
    @property
    def days_until_deadline(self) -> int:
        return (self.target_start_date - TODAY).days
    
    @property
    def urgency_score(self) -> float:
        return self.days_until_deadline - self.avg_days_to_hire
    
    @property
    def gamma_shape(self) -> float:
        return COMPLEXITY_SHAPE_K.get(self.complexity, 2.5)
    
    @property
    def gamma_scale(self) -> float:
        return self.avg_days_to_hire / self.gamma_shape


@dataclass
class SimulationResult:
    role_id: str
    role_name: str
    recruiter_name: str
    recruiter_id: str
    target_date: date
    complexity: str
    urgency_score: float
    mean_completion_days: float
    std_completion_days: float
    p10_completion_days: float
    p50_completion_days: float
    p90_completion_days: float
    on_time_probability: float
    on_time_ci_lower: float
    on_time_ci_upper: float
