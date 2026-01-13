"""LLM integration using OpenRouter for AI-powered recommendations."""

import os
import json
from typing import Dict, List, Optional
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_ai_recommendations(
    results: Dict,
    bottlenecks: List[Dict],
    recruiters: Dict,
    api_key: Optional[str] = None,
    model: str = "anthropic/claude-3.5-sonnet"
) -> Optional[str]:
    """
    Generate AI-powered recommendations using OpenRouter.
    
    Args:
        results: Analysis results from simulation
        bottlenecks: Identified bottlenecks
        recruiters: Recruiter data
        api_key: OpenRouter API key (or from env OPENROUTER_API_KEY)
        model: Model to use (default: claude-3.5-sonnet)
    
    Returns:
        AI-generated recommendations as markdown string, or None if failed
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        return None
    
    # Prepare context for the LLM
    context = _build_context(results, bottlenecks, recruiters)
    
    prompt = f"""You are a hiring operations expert. Analyze this hiring plan data and provide actionable recommendations.

## Current Situation

{context}

## Your Task

Based on this data, provide:

1. **Executive Summary** (2-3 sentences)
2. **Top 3 Priority Actions** - Specific, actionable steps with expected impact
3. **Risk Mitigation** - How to handle the highest-risk roles
4. **Resource Optimization** - How to better distribute workload
5. **Timeline Adjustments** - Which deadlines should be reconsidered

Be specific. Reference actual role IDs, recruiter names, and numbers from the data.
Format your response in clean markdown."""

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://hiring-feasibility-engine.local",
                "X-Title": "Hiring Feasibility Engine"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return None
            
    except Exception:
        return None


def _build_context(results: Dict, bottlenecks: List[Dict], recruiters: Dict) -> str:
    """Build context string for the LLM prompt."""
    
    # Overall metrics
    context = f"""### Overall Metrics
- **Success Rate:** {results['overall_success_rate']*100:.1f}%
- **Critical Failure Risk:** {results['critical_failure_prob']*100:.1f}% (probability of >5 missed deadlines)
- **Expected Missed Deadlines:** {results['expected_missed']:.1f}
- **Worst Case (P99):** {results['p99_missed']:.0f} missed deadlines

### Role Summary
- Total Roles: {len(results['role_analyses'])}
- High Risk (<50% success): {sum(1 for r in results['role_analyses'] if r.on_time_probability < 0.5)}
- At Risk (50-75% success): {sum(1 for r in results['role_analyses'] if 0.5 <= r.on_time_probability < 0.75)}
- On Track (≥75% success): {sum(1 for r in results['role_analyses'] if r.on_time_probability >= 0.75)}

### Recruiter Workload
"""
    
    # Recruiter info
    for rec_id, rec in recruiters.items():
        rec_roles = [r for r in results['role_analyses'] if r.recruiter_id == rec_id]
        if rec_roles:
            avg_success = sum(r.on_time_probability for r in rec_roles) / len(rec_roles)
            high_risk = sum(1 for r in rec_roles if r.on_time_probability < 0.5)
            context += f"- **{rec.name}**: {len(rec_roles)} roles, {avg_success*100:.0f}% avg success, {high_risk} high-risk\n"
    
    # Bottlenecks
    if bottlenecks:
        context += "\n### Identified Bottlenecks\n"
        for b in bottlenecks[:5]:
            context += f"- **{b['recruiter']}**: {b['failure_ratio']:.1f}x failure rate, worst role: {b['worst_role'].role_id} ({b['worst_role'].on_time_probability*100:.0f}% success)\n"
    
    # Highest risk roles
    high_risk_roles = sorted(results['role_analyses'], key=lambda r: r.on_time_probability)[:5]
    context += "\n### Highest Risk Roles\n"
    for r in high_risk_roles:
        context += f"- **{r.role_id}** ({r.role_name}): {r.on_time_probability*100:.0f}% success, assigned to {r.recruiter_name}, due {r.target_date}\n"
    
    return context


def get_available_models() -> List[Dict]:
    """Return list of recommended models for this use case."""
    return [
        {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet (Recommended)"},
        {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku (Fast)"},
        {"id": "openai/gpt-4o", "name": "GPT-4o"},
        {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini (Fast)"},
        {"id": "google/gemini-pro-1.5", "name": "Gemini Pro 1.5"},
        {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B"},
    ]
