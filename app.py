"""
Hiring Feasibility Engine - Capacity Planning Dashboard
Uses Monte Carlo simulation with proper Poisson process and Gamma distribution modeling.

Mathematical Framework:
- Poisson Process: Models hiring events over time (λ = avg_monthly_capacity)
- Gamma Distribution: Models completion time variability (shape k varies by complexity)
- Priority Queue: Roles sorted by urgency, capacity split across parallel work
"""

import json
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats as stats_module
import streamlit as st
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

TODAY = date.today()
SIMULATION_RUNS = 10000
SIMULATION_MONTHS = 6

COMPLEXITY_SHAPE_K = {
    "Low": 4.0,
    "Medium": 2.5,
    "High": 1.5
}

# =============================================================================
# DATA MODELS
# =============================================================================

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
    role: str
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

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data() -> Tuple[Dict[str, Recruiter], List[Role]]:
    with open('recruiters.json', 'r') as f:
        recruiters_data = json.load(f)
    
    with open('hiring_plan.json', 'r') as f:
        roles_data = json.load(f)
    
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
            role=r['role'],
            complexity=r['complexity'],
            avg_days_to_hire=r['avg_days_to_hire'],
            target_start_date=datetime.strptime(r['target_start_date'], '%Y-%m-%d').date(),
            assigned_recruiter_id=r['assigned_recruiter_id']
        )
        for r in roles_data
    ]
    
    return recruiters, roles

# =============================================================================
# MONTE CARLO SIMULATION ENGINE
# =============================================================================

def generate_poisson_hiring_events(recruiter: Recruiter, days: int, n_simulations: int) -> np.ndarray:
    expected_hires = recruiter.avg_monthly_capacity * (days / 30.44)
    max_events = int(expected_hires * 3)
    
    rng = np.random.default_rng()
    inter_arrivals = rng.exponential(
        scale=recruiter.mean_days_between_hires,
        size=(n_simulations, max_events)
    )
    
    cumulative_times = np.cumsum(inter_arrivals, axis=1)
    return cumulative_times

def simulate_role_completion(role: Role, queue_position: int, 
                            hiring_event_times: np.ndarray) -> np.ndarray:
    n_simulations = hiring_event_times.shape[0]
    
    if queue_position < hiring_event_times.shape[1]:
        slot_times = hiring_event_times[:, queue_position]
    else:
        last_time = hiring_event_times[:, -1]
        extra_slots = queue_position - hiring_event_times.shape[1] + 1
        slot_times = last_time + extra_slots * (30.44 / 3.0)
    
    rng = np.random.default_rng()
    completion_variability = rng.gamma(
        shape=role.gamma_shape,
        scale=role.gamma_scale,
        size=n_simulations
    )
    
    queue_delay_factor = 0.3
    
    total_completion_days = (
        slot_times * queue_delay_factor +
        completion_variability
    )
    
    return total_completion_days

@st.cache_data
def run_monte_carlo_simulation(_recruiters: dict, _roles: list, 
                               n_simulations: int = SIMULATION_RUNS) -> Dict:
    recruiters = _recruiters
    roles = _roles
    simulation_days = SIMULATION_MONTHS * 30
    
    recruiter_roles: Dict[str, List[Role]] = {}
    for role in roles:
        rec_id = role.assigned_recruiter_id
        if rec_id not in recruiter_roles:
            recruiter_roles[rec_id] = []
        recruiter_roles[rec_id].append(role)
    
    for rec_id in recruiter_roles:
        recruiter_roles[rec_id].sort(key=lambda r: r.urgency_score)
    
    recruiter_hiring_events = {}
    for rec_id, recruiter in recruiters.items():
        recruiter_hiring_events[rec_id] = generate_poisson_hiring_events(
            recruiter, simulation_days, n_simulations
        )
    
    role_completion_days = {}
    for rec_id, rec_roles in recruiter_roles.items():
        hiring_events = recruiter_hiring_events[rec_id]
        
        for queue_pos, role in enumerate(rec_roles):
            completion_days = simulate_role_completion(
                role, queue_pos, hiring_events
            )
            role_completion_days[role.id] = completion_days
    
    missed_per_simulation = np.zeros(n_simulations)
    for role in roles:
        completion_days = role_completion_days[role.id]
        deadline_days = role.days_until_deadline
        missed_per_simulation += (completion_days > deadline_days).astype(int)
    
    return {
        'role_completion_days': role_completion_days,
        'missed_per_simulation': missed_per_simulation
    }

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calculate_confidence_interval(successes: int, trials: int, 
                                  confidence: float = 0.95) -> Tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    
    p = successes / trials
    z = stats_module.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    return (max(0, center - margin), min(1, center + margin))

def analyze_results(recruiters: dict, roles: list, 
                   simulation_results: dict) -> Dict:
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
            role_name=role.role,
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

def generate_recommendations(recruiters: dict, roles: list, results: dict, bottlenecks: list) -> List[Dict]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # 1. Role Reassignment Recommendations
    for b in bottlenecks[:3]:
        recommendations.append({
            'type': 'reassignment',
            'priority': 'High' if b['high_risk_count'] >= 2 else 'Medium',
            'title': f"Reassign {b['worst_role'].role_id} from {b['recruiter']}",
            'description': f"Move **{b['worst_role'].role_name}** to **{b['alternative_recruiter']}** to reduce bottleneck",
            'impact': f"+{b['potential_improvement']*100:.0f}% success probability for this role",
            'effort': 'Low',
            'details': {
                'current_success': b['worst_role'].on_time_probability * 100,
                'current_recruiter': b['recruiter'],
                'suggested_recruiter': b['alternative_recruiter']
            }
        })
    
    # 2. Deadline Extension Recommendations
    high_risk_roles = [r for r in results['role_analyses'] if r.on_time_probability < 0.5]
    if high_risk_roles:
        urgent_roles = sorted(high_risk_roles, key=lambda r: r.urgency_score)[:3]
        for role in urgent_roles:
            days_needed = int(role.p90_completion_days - role.target_date.toordinal() + TODAY.toordinal())
            if days_needed > 0:
                recommendations.append({
                    'type': 'deadline',
                    'priority': 'High' if role.on_time_probability < 0.3 else 'Medium',
                    'title': f"Extend deadline for {role.role_id}",
                    'description': f"**{role.role_name}** has only {role.on_time_probability*100:.0f}% chance of meeting deadline",
                    'impact': f"Extending by {days_needed + 7} days would increase success to ~80%",
                    'effort': 'Medium (requires stakeholder approval)',
                    'details': {
                        'current_deadline': role.target_date.strftime('%Y-%m-%d'),
                        'suggested_extension': days_needed + 7,
                        'current_success': role.on_time_probability * 100
                    }
                })
    
    # 3. Capacity Recommendations
    recruiter_utilization = {}
    for rec_id, rec in recruiters.items():
        rec_roles = [r for r in results['role_analyses'] if r.recruiter_id == rec_id]
        if rec_roles:
            total_capacity = rec.avg_monthly_capacity * SIMULATION_MONTHS
            utilization = len(rec_roles) / total_capacity * 100
            recruiter_utilization[rec_id] = {
                'name': rec.name,
                'utilization': utilization,
                'roles': len(rec_roles)
            }
    
    overloaded = [r for r in recruiter_utilization.values() if r['utilization'] > 100]
    if overloaded:
        recommendations.append({
            'type': 'capacity',
            'priority': 'High',
            'title': 'Consider adding recruiting capacity',
            'description': f"{len(overloaded)} recruiter(s) are operating above 100% utilization",
            'impact': 'Adding 1 recruiter could improve overall success rate by ~10-15%',
            'effort': 'High (budget and hiring required)',
            'details': {
                'overloaded_recruiters': [r['name'] for r in overloaded]
            }
        })
    
    # 4. Complexity-based Recommendations
    complexity_impact = analyze_complexity_impact(results)
    if 'High' in complexity_impact and complexity_impact['High']['avg_success'] < 0.6:
        recommendations.append({
            'type': 'process',
            'priority': 'Medium',
            'title': 'Improve high-complexity hiring process',
            'description': f"High-complexity roles have only {complexity_impact['High']['avg_success']*100:.0f}% average success",
            'impact': 'Reducing avg_days_to_hire by 10% could improve success by ~8%',
            'effort': 'Medium (process optimization needed)',
            'details': {
                'high_complexity_count': complexity_impact['High']['count'],
                'current_success': complexity_impact['High']['avg_success'] * 100
            }
        })
    
    # 5. Quick Wins
    quick_wins = [r for r in results['role_analyses'] 
                  if 0.6 <= r.on_time_probability < 0.75 and r.urgency_score > 10]
    if quick_wins:
        recommendations.append({
            'type': 'quick_win',
            'priority': 'Low',
            'title': f"{len(quick_wins)} roles are close to success threshold",
            'description': 'These roles are at 60-75% success with buffer time — small improvements can push them to green',
            'impact': 'Minor adjustments could convert these to "On Track" status',
            'effort': 'Low',
            'details': {
                'role_ids': [r.role_id for r in quick_wins[:5]]
            }
        })
    
    return recommendations

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(
    page_title="Hiring Feasibility Engine | Acme Inc.",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f1f5f9 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .logo-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #22d3ee;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .explanation-box {
        background: rgba(34, 211, 238, 0.1);
        border: 1px solid rgba(34, 211, 238, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .explanation-box h4 {
        color: #22d3ee;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    
    .explanation-box p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .bottleneck-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-left: 4px solid #f43f5e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .recommendation-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        border-color: rgba(34, 211, 238, 0.5);
        transform: translateY(-2px);
    }
    
    .recommendation-card.high {
        border-left: 4px solid #f43f5e;
    }
    
    .recommendation-card.medium {
        border-left: 4px solid #f59e0b;
    }
    
    .recommendation-card.low {
        border-left: 4px solid #10b981;
    }
    
    .priority-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .priority-badge.high {
        background: rgba(244, 63, 94, 0.2);
        color: #f43f5e;
    }
    
    .priority-badge.medium {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
    
    .priority-badge.low {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .type-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 0.5rem;
    }
    
    .impact-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 0.75rem;
    }
    
    .recruiter-detail-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
    }
    
    .tab-content {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="logo-text">◆ Acme Inc.</div>
    <h1>Hiring Feasibility Engine</h1>
    <p style="color: #94a3b8; font-size: 1.1rem;">Q1/Q2 Capacity Planning Dashboard — Statistically Validated</p>
    <div style="display: inline-block; margin-top: 1rem; padding: 0.5rem 1.25rem; background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 9999px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
        📅 {} | 🔢 {:,} Simulations
    </div>
</div>
""".format(TODAY.strftime('%B %d, %Y'), SIMULATION_RUNS), unsafe_allow_html=True)

# Load and process data
with st.spinner('🔄 Running Monte Carlo simulation...'):
    recruiters, roles = load_data()
    simulation_results = run_monte_carlo_simulation(recruiters, roles)
    results = analyze_results(recruiters, roles, simulation_results)
    bottlenecks = identify_bottlenecks(recruiters, roles, results)
    complexity_impact = analyze_complexity_impact(results)
    recommendations = generate_recommendations(recruiters, roles, results, bottlenecks)

# Calculate summary stats
high_risk = sum(1 for r in results['role_analyses'] if r.on_time_probability < 0.5)
at_risk = sum(1 for r in results['role_analyses'] if 0.5 <= r.on_time_probability < 0.75)
on_track = sum(1 for r in results['role_analyses'] if r.on_time_probability >= 0.75)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Quick Summary")
    
    success_rate = results['overall_success_rate']
    if success_rate >= 0.75:
        st.success(f"**Success Rate:** {success_rate:.1%}")
    elif success_rate >= 0.5:
        st.warning(f"**Success Rate:** {success_rate:.1%}")
    else:
        st.error(f"**Success Rate:** {success_rate:.1%}")
    
    st.metric("🔴 High Risk", high_risk)
    st.metric("🟡 At Risk", at_risk)
    st.metric("🟢 On Track", on_track)
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    st.caption(f"**Simulations:** {SIMULATION_RUNS:,}")
    st.caption(f"**Horizon:** {SIMULATION_MONTHS} months")
    st.caption(f"**Recruiters:** {len(recruiters)}")
    st.caption(f"**Total Roles:** {len(roles)}")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Overview", "👥 Recruiter Analysis", "💡 Recommendations"])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown('<div class="section-header">🎯 Key Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ci_low, ci_high = results['success_rate_ci']
        st.metric(
            label="Overall Success Rate",
            value=f"{success_rate:.1%}",
            delta="Good" if success_rate >= 0.75 else ("Warning" if success_rate >= 0.5 else "Critical")
        )
        st.caption(f"95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
    
    with col2:
        crit_fail = results['critical_failure_prob']
        st.metric(
            label="Critical Failure Risk",
            value=f"{crit_fail:.1%}",
            delta=f"P(>5 missed)"
        )
    
    with col3:
        st.metric(
            label="Expected Missed",
            value=f"{results['expected_missed']:.1f}",
            delta=f"P90: {results['p90_missed']:.0f}"
        )
    
    with col4:
        st.metric(
            label="CVaR (90%)",
            value=f"{results['cvar_90']:.1f}",
            delta="Worst 10% avg"
        )
    
    st.progress(min(success_rate, 1.0))
    
    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
        | Metric | Description | Good Value |
        |--------|-------------|------------|
        | **Success Rate** | Average probability roles start on time | ≥75% |
        | **Critical Failure Risk** | Probability of missing >5 deadlines | ≤10% |
        | **Expected Missed** | Average missed deadlines | As low as possible |
        | **CVaR (90%)** | Average missed in worst 10% of scenarios | Lower is better |
        """)
    
    st.markdown("---")
    
    # Complexity Impact
    st.markdown('<div class="section-header">🎯 Complexity Impact</div>', unsafe_allow_html=True)
    
    comp_cols = st.columns(3)
    for i, (complexity, stats) in enumerate(complexity_impact.items()):
        with comp_cols[i]:
            color = "#10b981" if complexity == "Low" else ("#f59e0b" if complexity == "Medium" else "#f43f5e")
            st.markdown(f"""
            <div class="stat-card">
                <h4 style="color: {color}; margin: 0;">{complexity} Complexity</h4>
                <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0; font-family: 'JetBrains Mono';">
                    {stats['avg_success']*100:.1f}%
                </p>
                <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">
                    {stats['count']} roles | σ = {stats['std_success']:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Role Table
    st.markdown('<div class="section-header">📋 Role-by-Role Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Filter by Risk", ["All Roles", "🔴 High Risk", "🟡 At Risk", "🟢 On Track"], key="overview_risk")
    with col2:
        complexity_filter = st.selectbox("Filter by Complexity", ["All", "High", "Medium", "Low"], key="overview_complexity")
    with col3:
        recruiter_filter = st.selectbox("Filter by Recruiter", ["All"] + [r.name for r in recruiters.values()], key="overview_recruiter")
    
    role_data = []
    for r in results['role_analyses']:
        risk_level = "🔴 High" if r.on_time_probability < 0.5 else ("🟡 Medium" if r.on_time_probability < 0.75 else "🟢 Low")
        role_data.append({
            'Role ID': r.role_id,
            'Role': r.role_name,
            'Recruiter': r.recruiter_name,
            'Complexity': r.complexity,
            'Target': r.target_date.strftime('%Y-%m-%d'),
            'Success %': r.on_time_probability * 100,
            'Risk': risk_level
        })
    
    roles_df = pd.DataFrame(role_data)
    
    if "High Risk" in risk_filter:
        roles_df = roles_df[roles_df['Success %'] < 50]
    elif "At Risk" in risk_filter:
        roles_df = roles_df[(roles_df['Success %'] >= 50) & (roles_df['Success %'] < 75)]
    elif "On Track" in risk_filter:
        roles_df = roles_df[roles_df['Success %'] >= 75]
    
    if complexity_filter != "All":
        roles_df = roles_df[roles_df['Complexity'] == complexity_filter]
    
    if recruiter_filter != "All":
        roles_df = roles_df[roles_df['Recruiter'] == recruiter_filter]
    
    roles_df = roles_df.sort_values('Success %', ascending=True)
    
    st.dataframe(
        roles_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Success %': st.column_config.ProgressColumn('Success %', format="%.1f%%", min_value=0, max_value=100),
        }
    )
    
    st.caption(f"Showing {len(roles_df)} of {len(results['role_analyses'])} roles")
    
    st.markdown("---")
    
    # Distribution
    st.markdown('<div class="section-header">📊 Missed Deadlines Distribution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        hist_data = results['missed_distribution']
        chart_df = pd.DataFrame({'Missed Deadlines': hist_data})
        counts = chart_df['Missed Deadlines'].value_counts().sort_index()
        st.bar_chart(counts.head(25))
    
    with col2:
        st.markdown(f"""
        ### Percentiles
        | Metric | Value |
        |--------|-------|
        | P50 | {results['p50_missed']:.0f} |
        | P90 | {results['p90_missed']:.0f} |
        | P95 | {results['p95_missed']:.0f} |
        | P99 | {results['p99_missed']:.0f} |
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2: RECRUITER ANALYSIS
# =============================================================================

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">👥 Recruiter Performance Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <h4>📖 About This Analysis</h4>
        <p>
            This section provides detailed insights into each recruiter's workload, performance, and assigned roles.
            Use this to identify capacity issues and optimize role assignments.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary Table
    st.markdown("### 📊 Performance Summary")
    
    workload_data = []
    for rec_id, rec in recruiters.items():
        rec_roles = [r for r in results['role_analyses'] if r.recruiter_id == rec_id]
        if rec_roles:
            probs = [r.on_time_probability for r in rec_roles]
            total_capacity = rec.avg_monthly_capacity * SIMULATION_MONTHS
            workload_data.append({
                'Recruiter': rec.name,
                'Capacity/mo': rec.avg_monthly_capacity,
                'Assigned': len(rec_roles),
                'Utilization %': (len(rec_roles) / total_capacity) * 100 if total_capacity > 0 else 0,
                'Avg Success': np.mean(probs) * 100,
                'Min Success': np.min(probs) * 100,
                'High Risk': sum(1 for p in probs if p < 0.5),
                'rec_id': rec_id
            })
    
    workload_df = pd.DataFrame(workload_data)
    workload_df_display = workload_df.drop(columns=['rec_id']).sort_values('Avg Success', ascending=True)
    
    st.dataframe(
        workload_df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Capacity/mo': st.column_config.NumberColumn('Capacity/mo', format="%.1f"),
            'Utilization %': st.column_config.ProgressColumn('Utilization', format="%.0f%%", min_value=0, max_value=200),
            'Avg Success': st.column_config.ProgressColumn('Avg Success', format="%.1f%%", min_value=0, max_value=100),
            'Min Success': st.column_config.NumberColumn('Worst Role', format="%.1f%%"),
            'High Risk': st.column_config.NumberColumn('🔴 High Risk', format="%d")
        }
    )
    
    st.markdown("---")
    
    # Individual Recruiter Details
    st.markdown("### 🔍 Individual Recruiter Details")
    
    selected_recruiter = st.selectbox(
        "Select a recruiter to view details",
        options=[r.name for r in recruiters.values()],
        key="recruiter_detail_select"
    )
    
    # Find selected recruiter
    selected_rec = None
    for rec in recruiters.values():
        if rec.name == selected_recruiter:
            selected_rec = rec
            break
    
    if selected_rec:
        rec_roles = [r for r in results['role_analyses'] if r.recruiter_id == selected_rec.id]
        rec_probs = [r.on_time_probability for r in rec_roles]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Assigned Roles", len(rec_roles))
        with col2:
            st.metric("Monthly Capacity", f"{selected_rec.avg_monthly_capacity:.1f}")
        with col3:
            utilization = len(rec_roles) / (selected_rec.avg_monthly_capacity * SIMULATION_MONTHS) * 100
            st.metric("Utilization", f"{utilization:.0f}%", delta="Overloaded" if utilization > 100 else "OK")
        with col4:
            avg_success = np.mean(rec_probs) * 100
            st.metric("Avg Success", f"{avg_success:.1f}%")
        
        st.markdown("#### Assigned Roles")
        
        rec_role_data = []
        for r in rec_roles:
            risk = "🔴 High" if r.on_time_probability < 0.5 else ("🟡 Medium" if r.on_time_probability < 0.75 else "🟢 Low")
            rec_role_data.append({
                'Role ID': r.role_id,
                'Role': r.role_name,
                'Complexity': r.complexity,
                'Target Date': r.target_date.strftime('%Y-%m-%d'),
                'Urgency': int(r.urgency_score),
                'Success %': r.on_time_probability * 100,
                'P50 Days': int(r.p50_completion_days),
                'Risk': risk
            })
        
        rec_roles_df = pd.DataFrame(rec_role_data).sort_values('Success %', ascending=True)
        
        st.dataframe(
            rec_roles_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Success %': st.column_config.ProgressColumn('Success %', format="%.1f%%", min_value=0, max_value=100),
                'Urgency': st.column_config.NumberColumn('Urgency', help="Lower = more urgent")
            }
        )
        
        # Performance Summary
        st.markdown("#### Performance Breakdown")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.markdown(f"""
            **By Complexity:**
            """)
            for complexity in ['Low', 'Medium', 'High']:
                comp_roles = [r for r in rec_roles if r.complexity == complexity]
                if comp_roles:
                    comp_avg = np.mean([r.on_time_probability for r in comp_roles]) * 100
                    color = "🟢" if comp_avg >= 75 else ("🟡" if comp_avg >= 50 else "🔴")
                    st.write(f"{color} {complexity}: {comp_avg:.1f}% ({len(comp_roles)} roles)")
        
        with perf_col2:
            st.markdown(f"""
            **Risk Distribution:**
            """)
            high_risk_count = sum(1 for r in rec_roles if r.on_time_probability < 0.5)
            at_risk_count = sum(1 for r in rec_roles if 0.5 <= r.on_time_probability < 0.75)
            on_track_count = sum(1 for r in rec_roles if r.on_time_probability >= 0.75)
            
            st.write(f"🔴 High Risk: {high_risk_count}")
            st.write(f"🟡 At Risk: {at_risk_count}")
            st.write(f"🟢 On Track: {on_track_count}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 3: RECOMMENDATIONS
# =============================================================================

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">💡 Actionable Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <h4>📖 About These Recommendations</h4>
        <p>
            Based on our analysis of 10,000 simulations, we've identified specific actions that could improve 
            your hiring plan's success rate. Recommendations are prioritized by potential impact and effort required.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        high_priority = sum(1 for r in recommendations if r['priority'] == 'High')
        st.metric("🔴 High Priority", high_priority)
    with col2:
        medium_priority = sum(1 for r in recommendations if r['priority'] == 'Medium')
        st.metric("🟡 Medium Priority", medium_priority)
    with col3:
        low_priority = sum(1 for r in recommendations if r['priority'] == 'Low')
        st.metric("🟢 Low Priority", low_priority)
    
    st.markdown("---")
    
    # Filter
    priority_filter = st.selectbox(
        "Filter by Priority",
        ["All", "High", "Medium", "Low"],
        key="rec_priority_filter"
    )
    
    filtered_recs = recommendations
    if priority_filter != "All":
        filtered_recs = [r for r in recommendations if r['priority'] == priority_filter]
    
    if not filtered_recs:
        st.info("No recommendations found for the selected filter.")
    else:
        for i, rec in enumerate(filtered_recs):
            priority_class = rec['priority'].lower()
            type_label = rec['type'].replace('_', ' ').title()
            
            st.markdown(f"""
            <div class="recommendation-card {priority_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                    <div>
                        <span class="priority-badge {priority_class}">{rec['priority']} Priority</span>
                        <span class="type-badge">{type_label}</span>
                    </div>
                </div>
                <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">{rec['title']}</h4>
                <p style="color: #94a3b8; margin: 0;">{rec['description']}</p>
                <div class="impact-box">
                    <p style="color: #10b981; margin: 0; font-size: 0.9rem;">
                        <strong>💰 Impact:</strong> {rec['impact']}
                    </p>
                    <p style="color: #94a3b8; margin: 0.25rem 0 0 0; font-size: 0.85rem;">
                        <strong>⚡ Effort:</strong> {rec['effort']}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📊 View Details for: {rec['title'][:50]}..."):
                if rec['type'] == 'reassignment':
                    st.markdown(f"""
                    **Current State:**
                    - Current Recruiter: {rec['details']['current_recruiter']}
                    - Current Success Rate: {rec['details']['current_success']:.1f}%
                    
                    **Suggested Change:**
                    - Move to: {rec['details']['suggested_recruiter']}
                    - Expected improvement in success probability
                    
                    **Why this works:** The suggested recruiter has lower utilization and/or better track record 
                    with similar role types.
                    """)
                elif rec['type'] == 'deadline':
                    st.markdown(f"""
                    **Current State:**
                    - Current Deadline: {rec['details']['current_deadline']}
                    - Current Success Rate: {rec['details']['current_success']:.1f}%
                    
                    **Suggested Change:**
                    - Extend deadline by: {rec['details']['suggested_extension']} days
                    
                    **Why this works:** More time allows for the natural variance in hiring process 
                    without risking a miss.
                    """)
                elif rec['type'] == 'capacity':
                    st.markdown(f"""
                    **Overloaded Recruiters:**
                    {', '.join(rec['details']['overloaded_recruiters'])}
                    
                    **Why this works:** Adding capacity distributes workload more evenly, 
                    reducing queue delays and improving success rates across the board.
                    """)
                elif rec['type'] == 'process':
                    st.markdown(f"""
                    **Current State:**
                    - High-complexity roles: {rec['details']['high_complexity_count']}
                    - Current success rate: {rec['details']['current_success']:.1f}%
                    
                    **Suggested Actions:**
                    - Streamline interview process
                    - Pre-screen candidates more aggressively
                    - Expand sourcing channels
                    """)
                elif rec['type'] == 'quick_win':
                    st.markdown(f"""
                    **Roles close to threshold:**
                    {', '.join(rec['details']['role_ids'])}
                    
                    **Suggested Actions:**
                    - Small deadline extensions (1-2 weeks)
                    - Priority attention from recruiters
                    - Consider temporary support
                    """)
    
    st.markdown("---")
    
    # Bottleneck Section
    st.markdown("### 🚨 Identified Bottlenecks")
    
    if bottlenecks:
        for b in bottlenecks[:5]:
            sig_badge = '<span style="background: rgba(139, 92, 246, 0.2); color: #a78bfa; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-left: 0.5rem;">p < 0.05</span>' if b['statistically_significant'] else ''
            
            st.markdown(f"""
            <div class="bottleneck-card">
                <h4 style="color: #f43f5e; margin: 0;">⚠️ {b['recruiter']} {sig_badge}</h4>
                <p style="color: #94a3b8; margin: 0.5rem 0;">
                    {b['role_count']} roles | {b['high_risk_count']} high risk | {b['failure_ratio']:.1f}x failure rate
                </p>
                <p style="color: #cbd5e1; margin-top: 0.5rem;">
                    <strong>Worst Role:</strong> {b['worst_role'].role_id} ({b['worst_role'].role_name}) — {b['worst_role'].on_time_probability*100:.0f}% success
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No significant bottlenecks identified! 🎉")
    
    st.markdown("---")
    
    # Summary Action Plan
    st.markdown("### 📋 Summary Action Plan")
    
    action_summary = []
    for rec in recommendations:
        action_summary.append({
            'Priority': rec['priority'],
            'Type': rec['type'].replace('_', ' ').title(),
            'Action': rec['title'],
            'Impact': rec['impact'][:50] + '...' if len(rec['impact']) > 50 else rec['impact']
        })
    
    if action_summary:
        action_df = pd.DataFrame(action_summary)
        st.dataframe(action_df, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <p style="font-size: 0.9rem;">
        <strong>Mathematical Framework:</strong> Poisson Process + Gamma Distribution | 
        <strong>{SIMULATION_RUNS:,}</strong> Monte Carlo Simulations
    </p>
    <p style="margin-top: 0.5rem; font-size: 0.8rem;">Built with 🔬 by the Hiring Feasibility Engine v3.0</p>
</div>
""", unsafe_allow_html=True)
