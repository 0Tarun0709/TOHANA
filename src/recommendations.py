"""Recommendation generation for the Hiring Feasibility Engine."""

from typing import Dict, List
from src.config import TODAY, SIMULATION_MONTHS
from src.analysis import analyze_complexity_impact


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
