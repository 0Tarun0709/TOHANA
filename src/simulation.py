"""Monte Carlo simulation engine for the Hiring Feasibility Engine."""

from typing import Dict, List
import numpy as np
from src.models import Role, Recruiter
from src.config import SIMULATION_RUNS


def simulate_role_completion(role: Role, queue_position: int, 
                            recruiter: Recruiter, n_simulations: int,
                            seed: int = None) -> np.ndarray:
    """Simulate completion times for a role."""
    rng = np.random.default_rng(seed)
    
    # Queue wait time based on position and recruiter capacity
    monthly_capacity = recruiter.avg_monthly_capacity
    expected_wait_months = queue_position / monthly_capacity
    expected_wait_days = expected_wait_months * 30.44
    
    if expected_wait_days > 1:
        wait_shape = 2.0
        wait_scale = expected_wait_days / wait_shape
        queue_wait = rng.gamma(shape=wait_shape, scale=wait_scale, size=n_simulations)
    else:
        queue_wait = np.zeros(n_simulations)
    
    # Process time with variability based on complexity
    process_shape = role.gamma_shape
    process_scale = role.avg_days_to_hire / process_shape
    process_time = rng.gamma(shape=process_shape, scale=process_scale, size=n_simulations)
    
    return queue_wait + process_time


def run_monte_carlo_simulation(recruiters: dict, roles: list, 
                               n_simulations: int = SIMULATION_RUNS) -> Dict:
    """Run Monte Carlo simulation for the entire hiring plan."""
    # Group roles by recruiter and sort by urgency
    recruiter_roles: Dict[str, List[Role]] = {}
    for role in roles:
        rec_id = role.assigned_recruiter_id
        if rec_id not in recruiter_roles:
            recruiter_roles[rec_id] = []
        recruiter_roles[rec_id].append(role)
    
    for rec_id in recruiter_roles:
        recruiter_roles[rec_id].sort(key=lambda r: r.urgency_score)
    
    # Simulate each role
    role_completion_days = {}
    base_seed = 42
    
    for rec_id, rec_roles in recruiter_roles.items():
        recruiter = recruiters[rec_id]
        
        for queue_pos, role in enumerate(rec_roles):
            role_seed = base_seed + hash(role.id) % 10000
            
            completion_days = simulate_role_completion(
                role=role,
                queue_position=queue_pos,
                recruiter=recruiter,
                n_simulations=n_simulations,
                seed=role_seed
            )
            role_completion_days[role.id] = completion_days
    
    # Count missed deadlines per simulation
    missed_per_simulation = np.zeros(n_simulations)
    for role in roles:
        completion_days = role_completion_days[role.id]
        deadline_days = role.days_until_deadline
        missed_per_simulation += (completion_days > deadline_days).astype(int)
    
    return {
        'role_completion_days': role_completion_days,
        'missed_per_simulation': missed_per_simulation
    }
