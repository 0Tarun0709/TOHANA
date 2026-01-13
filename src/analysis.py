"""Statistical analysis for the Hiring Feasibility Engine."""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats as stats_module
from src.models import SimulationResult
from src.config import SIMULATION_RUNS, SIMULATION_MONTHS


def calculate_confidence_interval(successes: int, trials: int, 
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion."""
    if trials == 0:
        return (0.0, 1.0)
    
    p = successes / trials
    z = stats_module.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def analyze_results(recruiters: dict, roles: list, simulation_results: dict) -> Dict:
    """Analyze simulation results and compute statistics."""
    role_completion_days = simulation_results['role_completion_days']
    missed_per_simulation = simulation_results['missed_per_simulation']
    n_sims = SIMULATION_RUNS
    
    role_analyses = []
    on_time_probs = []
    
    for role in roles:
        completion_days = role_completion_days[role.id]
        deadline_days = role.days_until_deadline
        
        on_time_count = int(np.sum(completion_days <= deadline_days))
        on_time_prob = on_time_count / n_sims
        on_time_probs.append(on_time_prob)
        
        ci_lower, ci_upper = calculate_confidence_interval(on_time_count, n_sims)
        
        mean_completion = np.mean(completion_days)
        std_completion = np.std(completion_days)
        p10_completion = np.percentile(completion_days, 10)
        p50_completion = np.percentile(completion_days, 50)
        p90_completion = np.percentile(completion_days, 90)
        
        recruiter = recruiters[role.assigned_recruiter_id]
        
        role_analyses.append(SimulationResult(
            role_id=role.id,
            role_name=role.role_name,
            recruiter_name=recruiter.name,
            recruiter_id=recruiter.id,
            target_date=role.target_start_date,
            complexity=role.complexity,
            urgency_score=role.urgency_score,
            mean_completion_days=mean_completion,
            std_completion_days=std_completion,
            p10_completion_days=p10_completion,
            p50_completion_days=p50_completion,
            p90_completion_days=p90_completion,
            on_time_probability=on_time_prob,
            on_time_ci_lower=ci_lower,
            on_time_ci_upper=ci_upper
        ))
    
    overall_success_rate = np.mean(on_time_probs)
    success_rate_std = np.std(on_time_probs) / np.sqrt(len(roles))
    success_rate_ci = (
        overall_success_rate - 1.96 * success_rate_std,
        overall_success_rate + 1.96 * success_rate_std
    )
    
    critical_failures = np.sum(missed_per_simulation > 5)
    critical_failure_prob = critical_failures / n_sims
    critical_ci = calculate_confidence_interval(int(critical_failures), n_sims)
    
    expected_missed = np.mean(missed_per_simulation)
    missed_std = np.std(missed_per_simulation)
    p50_missed = np.percentile(missed_per_simulation, 50)
    p90_missed = np.percentile(missed_per_simulation, 90)
    p95_missed = np.percentile(missed_per_simulation, 95)
    p99_missed = np.percentile(missed_per_simulation, 99)
    
    var_90 = np.percentile(missed_per_simulation, 90)
    var_95 = np.percentile(missed_per_simulation, 95)
    
    worst_10_pct = missed_per_simulation[missed_per_simulation >= var_90]
    cvar_90 = np.mean(worst_10_pct) if len(worst_10_pct) > 0 else var_90
    
    return {
        'role_analyses': role_analyses,
        'overall_success_rate': overall_success_rate,
        'success_rate_ci': success_rate_ci,
        'critical_failure_prob': critical_failure_prob,
        'critical_failure_ci': critical_ci,
        'expected_missed': expected_missed,
        'missed_std': missed_std,
        'p50_missed': p50_missed,
        'p90_missed': p90_missed,
        'p95_missed': p95_missed,
        'p99_missed': p99_missed,
        'var_90': var_90,
        'var_95': var_95,
        'cvar_90': cvar_90,
        'missed_distribution': missed_per_simulation
    }


def identify_bottlenecks(recruiters: dict, roles: list, results: dict) -> List[Dict]:
    """Identify overloaded recruiters and high-risk assignments."""
    role_analyses = results['role_analyses']
    overall_avg = results['overall_success_rate']
    
    recruiter_stats = {}
    for result in role_analyses:
        rec_id = result.recruiter_id
        if rec_id not in recruiter_stats:
            recruiter_stats[rec_id] = {
                'name': result.recruiter_name,
                'roles': [],
                'probs': []
            }
        recruiter_stats[rec_id]['roles'].append(result)
        recruiter_stats[rec_id]['probs'].append(result.on_time_probability)
    
    bottlenecks = []
    for rec_id, stats in recruiter_stats.items():
        probs = stats['probs']
        recruiter_avg = np.mean(probs)
        
        if len(probs) >= 2:
            _, p_value = stats_module.ttest_1samp(probs, overall_avg)
            significant = p_value < 0.05 and recruiter_avg < overall_avg
        else:
            p_value = 1.0
            significant = False
        
        failure_rate = 1 - recruiter_avg
        overall_failure_rate = 1 - overall_avg
        failure_ratio = failure_rate / overall_failure_rate if overall_failure_rate > 0 else 1.0
        
        worst_role = min(stats['roles'], key=lambda r: r.on_time_probability)
        
        at_risk = sum(1 for p in probs if p < 0.7)
        high_risk = sum(1 for p in probs if p < 0.5)
        
        rec_loads = {r: len(recruiter_stats.get(r, {}).get('roles', [])) 
                    for r in recruiters.keys()}
        rec_avgs = {r: np.mean(recruiter_stats.get(r, {}).get('probs', [1.0])) 
                   for r in recruiters.keys()}
        
        current_rec_avgs = rec_avgs.copy()
        current_rec_loads = rec_loads.copy()
        best_alt_id = max(
            [r for r in recruiters.keys() if r != rec_id],
            key=lambda r, avgs=current_rec_avgs, loads=current_rec_loads: avgs.get(r, 0) - loads.get(r, 0) * 0.1
        )
        best_alternative = recruiters[best_alt_id]
        
        alt_avg = rec_avgs.get(best_alt_id, overall_avg)
        potential_improvement = (alt_avg - worst_role.on_time_probability) * 0.5
        
        if at_risk >= 2 or significant or recruiter_avg < 0.6:
            bottlenecks.append({
                'recruiter_id': rec_id,
                'recruiter': stats['name'],
                'role_count': len(stats['roles']),
                'at_risk_count': at_risk,
                'high_risk_count': high_risk,
                'avg_success_prob': recruiter_avg,
                'failure_ratio': failure_ratio,
                'p_value': p_value,
                'statistically_significant': significant,
                'worst_role': worst_role,
                'recommendation': f"Reassign **{worst_role.role_id}** ({worst_role.role_name}) to **{best_alternative.name}**",
                'potential_improvement': max(0, potential_improvement),
                'alternative_recruiter': best_alternative.name
            })
    
    bottlenecks.sort(key=lambda b: -b['failure_ratio'])
    return bottlenecks


def analyze_complexity_impact(results: dict) -> Dict:
    """Analyze the impact of role complexity on success probability."""
    role_analyses = results['role_analyses']
    
    complexity_stats = {}
    for complexity in ['Low', 'Medium', 'High']:
        roles = [r for r in role_analyses if r.complexity == complexity]
        if roles:
            probs = [r.on_time_probability for r in roles]
            complexity_stats[complexity] = {
                'count': len(roles),
                'avg_success': np.mean(probs),
                'std_success': np.std(probs),
                'min_success': np.min(probs),
                'max_success': np.max(probs)
            }
    
    return complexity_stats
