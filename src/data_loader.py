"""Data loading and validation for the Hiring Feasibility Engine."""

import json
from datetime import datetime
from typing import Dict, List, Tuple
from src.models import Recruiter, Role


def parse_uploaded_data(recruiters_json: str, hiring_plan_json: str) -> Tuple[Dict[str, Recruiter], List[Role]]:
    """Parse uploaded JSON data into Recruiter and Role objects."""
    recruiters_data = json.loads(recruiters_json)
    roles_data = json.loads(hiring_plan_json)
    
    recruiters = {
        r['id']: Recruiter(
            id=r['id'],
            name=r['name'],
            avg_monthly_capacity=r['avg_monthly_capacity']
        )
        for r in recruiters_data
    }
    
    roles = [
        Role(
            id=r['id'],
            role_name=r['role'],
            complexity=r['complexity'],
            avg_days_to_hire=r['avg_days_to_hire'],
            target_start_date=datetime.strptime(r['target_start_date'], '%Y-%m-%d').date(),
            assigned_recruiter_id=r['assigned_recruiter_id']
        )
        for r in roles_data
    ]
    
    return recruiters, roles


def validate_data(recruiters: Dict, roles: List) -> Tuple[bool, str]:
    """Validate uploaded data for consistency."""
    errors = []
    
    for role in roles:
        if role.assigned_recruiter_id not in recruiters:
            errors.append(f"Role {role.id} assigned to unknown recruiter {role.assigned_recruiter_id}")
    
    if not recruiters:
        errors.append("No recruiters found in uploaded file")
    if not roles:
        errors.append("No roles found in uploaded file")
    
    if errors:
        return False, "\n".join(errors)
    return True, f"✅ Loaded {len(recruiters)} recruiters and {len(roles)} roles"
