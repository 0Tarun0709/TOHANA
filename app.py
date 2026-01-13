"""
Hiring Feasibility Engine - Capacity Planning Dashboard
Main Streamlit application.
"""

import json
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st

from src.config import TODAY, SIMULATION_RUNS, SIMULATION_MONTHS
from src.data_loader import parse_uploaded_data, validate_data
from src.simulation import run_monte_carlo_simulation
from src.analysis import analyze_results, identify_bottlenecks, analyze_complexity_impact
from src.recommendations import generate_recommendations
from src.llm import get_ai_recommendations, get_available_models

# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Hiring Feasibility Engine | Acme Inc.",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLES
# =============================================================================

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
    
    .recommendation-card.high { border-left: 4px solid #f43f5e; }
    .recommendation-card.medium { border-left: 4px solid #f59e0b; }
    .recommendation-card.low { border-left: 4px solid #10b981; }
    
    .priority-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .priority-badge.high { background: rgba(244, 63, 94, 0.2); color: #f43f5e; }
    .priority-badge.medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .priority-badge.low { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    
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
    
    .section-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
    }
    
    .tab-content { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <div class="logo-text">◆ Acme Inc.</div>
    <h1>Hiring Feasibility Engine</h1>
    <p style="color: #94a3b8; font-size: 1.1rem;">Q1/Q2 Capacity Planning Dashboard</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FILE UPLOAD
# =============================================================================

with st.sidebar:
    st.markdown("### 📁 Upload Data Files")
    
    uploaded_recruiters = st.file_uploader(
        "Upload recruiters.json",
        type=['json'],
        help="JSON file containing recruiter data"
    )
    
    uploaded_hiring_plan = st.file_uploader(
        "Upload hiring_plan.json",
        type=['json'],
        help="JSON file containing roles data"
    )
    
    st.markdown("---")

# Check if files are uploaded
if uploaded_recruiters is None or uploaded_hiring_plan is None:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
        <h2 style="color: #f1f5f9; margin-bottom: 1rem;">Upload Your Hiring Data</h2>
        <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            Upload both JSON files using the sidebar to generate your hiring feasibility analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1.5rem;">
            <h4 style="color: #22d3ee; margin: 0 0 1rem 0;">📋 recruiters.json</h4>
            <pre style="background: rgba(15, 23, 42, 0.8); padding: 1rem; border-radius: 8px; font-size: 0.8rem; color: #e2e8f0;">
[
  {
    "id": "R_01",
    "name": "Sarah (Lead)",
    "avg_monthly_capacity": 4.5
  }
]</pre>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1.5rem;">
            <h4 style="color: #22d3ee; margin: 0 0 1rem 0;">📋 hiring_plan.json</h4>
            <pre style="background: rgba(15, 23, 42, 0.8); padding: 1rem; border-radius: 8px; font-size: 0.8rem; color: #e2e8f0;">
[
  {
    "id": "JOB_001",
    "role": "Backend Engineer",
    "complexity": "High",
    "avg_days_to_hire": 75,
    "target_start_date": "2026-04-15",
    "assigned_recruiter_id": "R_01"
  }
]</pre>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# =============================================================================
# PROCESS DATA
# =============================================================================

try:
    recruiters_json = uploaded_recruiters.read().decode('utf-8')
    hiring_plan_json = uploaded_hiring_plan.read().decode('utf-8')
    
    recruiters, roles = parse_uploaded_data(recruiters_json, hiring_plan_json)
    
    is_valid, validation_message = validate_data(recruiters, roles)
    
    if not is_valid:
        st.error(f"❌ Data Validation Failed:\n{validation_message}")
        st.stop()

except json.JSONDecodeError as e:
    st.error(f"❌ Invalid JSON format: {str(e)}")
    st.stop()
except KeyError as e:
    st.error(f"❌ Missing required field: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error processing files: {str(e)}")
    st.stop()

# Show info badge
st.markdown(f"""
<div style="display: inline-block; margin-bottom: 1rem; padding: 0.5rem 1.25rem; background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 9999px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
    📅 {TODAY.strftime('%B %d, %Y')} | 🔢 {SIMULATION_RUNS:,} Simulations | 👥 {len(recruiters)} Recruiters | 📋 {len(roles)} Roles
</div>
""", unsafe_allow_html=True)

# =============================================================================
# RUN SIMULATION
# =============================================================================

with st.spinner('🔄 Running Monte Carlo simulation...'):
    simulation_results = run_monte_carlo_simulation(recruiters, roles)
    results = analyze_results(recruiters, roles, simulation_results)
    bottlenecks = identify_bottlenecks(recruiters, roles, results)
    complexity_impact = analyze_complexity_impact(results)
    recommendations = generate_recommendations(recruiters, roles, results, bottlenecks)

# Calculate summary stats
high_risk = sum(1 for r in results['role_analyses'] if r.on_time_probability < 0.5)
at_risk = sum(1 for r in results['role_analyses'] if 0.5 <= r.on_time_probability < 0.75)
on_track = sum(1 for r in results['role_analyses'] if r.on_time_probability >= 0.75)

# Sidebar summary
with st.sidebar:
    st.markdown("### 📊 Analysis Results")
    
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

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Overview", "👥 Recruiter Analysis", "💡 Recommendations"])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================

with tab1:
    st.markdown('<div class="section-header">🎯 Key Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ci_low, ci_high = results['success_rate_ci']
        st.metric("Overall Success Rate", f"{success_rate:.1%}")
        st.caption(f"95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
    
    with col2:
        st.metric("Critical Failure Risk", f"{results['critical_failure_prob']:.1%}")
        st.caption("P(>5 missed)")
    
    with col3:
        st.metric("Expected Missed", f"{results['expected_missed']:.1f}")
        st.caption(f"P90: {results['p90_missed']:.0f}")
    
    with col4:
        st.metric("CVaR (90%)", f"{results['cvar_90']:.1f}")
        st.caption("Worst 10% avg")
    
    st.progress(min(success_rate, 1.0))
    
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
                <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0; font-family: 'JetBrains Mono';">{stats['avg_success']*100:.1f}%</p>
                <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">{stats['count']} roles</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Role Table
    st.markdown('<div class="section-header">📋 Role Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Filter by Risk", ["All Roles", "🔴 High Risk", "🟡 At Risk", "🟢 On Track"])
    with col2:
        complexity_filter = st.selectbox("Filter by Complexity", ["All", "High", "Medium", "Low"])
    with col3:
        recruiter_filter = st.selectbox("Filter by Recruiter", ["All"] + [r.name for r in recruiters.values()])
    
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

# =============================================================================
# TAB 2: RECRUITER ANALYSIS
# =============================================================================

with tab2:
    st.markdown('<div class="section-header">👥 Recruiter Performance</div>', unsafe_allow_html=True)
    
    # Summary Table
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
    st.markdown("### 🔍 Recruiter Details")
    
    selected_recruiter = st.selectbox("Select recruiter", [r.name for r in recruiters.values()])
    
    selected_rec = next((r for r in recruiters.values() if r.name == selected_recruiter), None)
    
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
            st.metric("Utilization", f"{utilization:.0f}%")
        with col4:
            st.metric("Avg Success", f"{np.mean(rec_probs) * 100:.1f}%")
        
        st.markdown("#### Assigned Roles")
        
        rec_role_data = []
        for r in rec_roles:
            risk = "🔴" if r.on_time_probability < 0.5 else ("🟡" if r.on_time_probability < 0.75 else "🟢")
            rec_role_data.append({
                'Role ID': r.role_id,
                'Role': r.role_name,
                'Complexity': r.complexity,
                'Target': r.target_date.strftime('%Y-%m-%d'),
                'Success %': r.on_time_probability * 100,
                'Risk': risk
            })
        
        rec_roles_df = pd.DataFrame(rec_role_data).sort_values('Success %', ascending=True)
        
        st.dataframe(
            rec_roles_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Success %': st.column_config.ProgressColumn('Success %', format="%.1f%%", min_value=0, max_value=100)
            }
        )

# =============================================================================
# TAB 3: RECOMMENDATIONS
# =============================================================================

with tab3:
    st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
    
    # AI Recommendations Section
    st.markdown("### 🤖 AI-Powered Analysis")
    
    with st.expander("Configure AI Recommendations", expanded=False):
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai/keys"
        )
        
        models = get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            options=[m["id"] for m in models],
            format_func=lambda x: next(m["name"] for m in models if m["id"] == x)
        )
    
    if api_key:
        if st.button("🚀 Generate AI Recommendations", type="primary"):
            with st.spinner("Generating AI insights..."):
                ai_response = get_ai_recommendations(
                    results=results,
                    bottlenecks=bottlenecks,
                    recruiters=recruiters,
                    api_key=api_key,
                    model=selected_model
                )
            
            if ai_response:
                st.markdown("""
                <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
                    <h4 style="color: #a78bfa; margin: 0 0 1rem 0;">🤖 AI Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(ai_response)
            else:
                st.error("Failed to generate AI recommendations. Please check your API key.")
    else:
        st.info("💡 Enter your OpenRouter API key above to get AI-powered recommendations tailored to your data.")
    
    st.markdown("---")
    
    # Standard Recommendations
    st.markdown("### 📋 Standard Recommendations")
    
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
    
    priority_filter = st.selectbox("Filter by Priority", ["All", "High", "Medium", "Low"])
    
    filtered_recs = recommendations
    if priority_filter != "All":
        filtered_recs = [r for r in recommendations if r['priority'] == priority_filter]
    
    if not filtered_recs:
        st.info("No recommendations found.")
    else:
        for rec in filtered_recs:
            priority_class = rec['priority'].lower()
            type_label = rec['type'].replace('_', ' ').title()
            
            st.markdown(f"""
            <div class="recommendation-card {priority_class}">
                <span class="priority-badge {priority_class}">{rec['priority']} Priority</span>
                <span class="type-badge">{type_label}</span>
                <h4 style="color: #f1f5f9; margin: 0.75rem 0 0.5rem 0;">{rec['title']}</h4>
                <p style="color: #94a3b8; margin: 0;">{rec['description']}</p>
                <div class="impact-box">
                    <p style="color: #10b981; margin: 0;"><strong>Impact:</strong> {rec['impact']}</p>
                    <p style="color: #94a3b8; margin: 0.25rem 0 0 0;"><strong>Effort:</strong> {rec['effort']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bottlenecks
    st.markdown("### 🚨 Bottlenecks")
    
    if bottlenecks:
        for b in bottlenecks[:5]:
            st.markdown(f"""
            <div class="bottleneck-card">
                <h4 style="color: #f43f5e; margin: 0;">⚠️ {b['recruiter']}</h4>
                <p style="color: #94a3b8; margin: 0.5rem 0;">
                    {b['role_count']} roles | {b['high_risk_count']} high risk | {b['failure_ratio']:.1f}x failure rate
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No significant bottlenecks identified!")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <p>Analysis based on <span style="color: #22d3ee;">{SIMULATION_RUNS:,}</span> Monte Carlo simulations</p>
    <p style="font-size: 0.8rem;">Hiring Feasibility Engine v3.0</p>
</div>
""", unsafe_allow_html=True)
