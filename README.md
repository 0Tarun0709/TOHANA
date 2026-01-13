# Hiring Feasibility Engine

A capacity planning dashboard that predicts whether your hiring plan will succeed using Monte Carlo simulation.

---

## Overview

Upload your recruiters and hiring plan, and the engine will:

- **Predict success rates** for each role
- **Identify bottlenecks** (overloaded recruiters)
- **Generate recommendations** to improve outcomes
- **Quantify risk** with statistical confidence

---

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -r requirements.txt
## or
uv sync
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

### 3. Upload Your Data

Upload two JSON files:
- `recruiters.json` - Your recruiting team
- `hiring_plan.json` - Roles to be filled

---

## Data Format

### recruiters.json

```json
[
  {
    "id": "R_01",
    "name": "Sarah (Lead)",
    "avg_monthly_capacity": 4.5
  },
  {
    "id": "R_02",
    "name": "Mike (Senior)",
    "avg_monthly_capacity": 3.5
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique recruiter identifier |
| `name` | string | Recruiter name |
| `avg_monthly_capacity` | number | Average hires completed per month |

### hiring_plan.json

```json
[
  {
    "id": "JOB_001",
    "role": "Staff Backend Engineer",
    "complexity": "High",
    "avg_days_to_hire": 75,
    "target_start_date": "2026-04-15",
    "assigned_recruiter_id": "R_01"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique job identifier |
| `role` | string | Job title |
| `complexity` | string | `Low`, `Medium`, or `High` |
| `avg_days_to_hire` | number | Historical average days to fill this role type |
| `target_start_date` | string | When the hire should start (YYYY-MM-DD) |
| `assigned_recruiter_id` | string | ID of assigned recruiter |

---

## Features

### 📈 Overview Tab

- **Overall Success Rate** - Probability that roles start on time
- **Critical Failure Risk** - Probability of missing >5 deadlines
- **Expected Missed Deadlines** - Average number of late starts
- **Complexity Impact** - How role complexity affects success
- **Role-by-Role Analysis** - Filterable table of all roles

### 👥 Recruiter Analysis Tab

- **Performance Summary** - All recruiters ranked by success rate
- **Utilization Tracking** - Who is overloaded?
- **Individual Details** - Drill down into each recruiter's workload

### 💡 Recommendations Tab

- **🤖 AI-Powered Analysis** - Get intelligent recommendations using OpenRouter
- **Prioritized Actions** - High/Medium/Low priority fixes
- **Reassignment Suggestions** - Move roles to better-suited recruiters
- **Deadline Extensions** - Which deadlines to push back
- **Capacity Alerts** - When to add recruiting resources
- **Bottleneck Analysis** - Who is the constraint?

---

## AI Recommendations (Optional)

The dashboard supports AI-powered recommendations using [OpenRouter](https://openrouter.ai/).

### Setup

1. Get an API key from [openrouter.ai/keys](https://openrouter.ai/keys)
2. Go to the **Recommendations** tab
3. Enter your API key
4. Select a model (Claude 3.5 Sonnet recommended)
5. Click **Generate AI Recommendations**


The AI analyzes your data and provides:
- Executive summary
- Priority actions with expected impact
- Risk mitigation strategies
- Resource optimization suggestions
- Timeline adjustment recommendations

---

## Project Structure

```
├── app.py                 # Streamlit dashboard (entry point)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── data/                  # Sample data files
│   ├── recruiters.json
│   └── hiring_plan.json
├── doc/                   # Documentation
│   └── METHODOLOGY.md     # Mathematical model explanation
└── src/                   # Source modules
    ├── __init__.py
    ├── config.py          # Constants (simulation runs, etc.)
    ├── models.py          # Data classes
    ├── data_loader.py     # JSON parsing & validation
    ├── simulation.py      # Monte Carlo engine
    ├── analysis.py        # Statistical analysis
    ├── recommendations.py # Recommendation generation
    └── llm.py             # OpenRouter AI integration
```

---

## How It Works

> 📖 For detailed math and methodology, see [doc/METHODOLOGY.md](doc/METHODOLOGY.md)

### Simulation Model

1. **Queue Wait Time**  
   Recruiters process roles based on capacity. A recruiter with capacity 3/month and 6 roles means later roles wait longer.

2. **Process Time**  
   Each role takes `avg_days_to_hire` on average, with variance based on complexity.

3. **Monte Carlo**  
   Run 5,000 simulations with random variation to get probability distributions.

### Risk Levels

| Level | Success Probability | Meaning |
|-------|---------------------|---------|
| 🟢 On Track | ≥ 75% | Likely to meet deadline |
| 🟡 At Risk | 50-75% | Needs monitoring |
| 🔴 High Risk | < 50% | Likely to miss deadline |

---

## Configuration

Edit `src/config.py` to adjust:

```python
SIMULATION_RUNS = 5000    # Number of Monte Carlo simulations
SIMULATION_MONTHS = 6     # Planning horizon

COMPLEXITY_SHAPE_K = {    # Variance by complexity
    "Low": 4.0,           # More predictable
    "Medium": 2.5,        
    "High": 1.5           # Less predictable
}
```

---

## Requirements

- Python 3.9+
- streamlit
- numpy
- scipy
- pandas
- requests (for AI recommendations)

---

