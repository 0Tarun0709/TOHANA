# Hiring Feasibility Engine - Methodology

This document explains the mathematical model and logic behind the simulation.

---

## Overview

The engine uses **Monte Carlo simulation** to predict whether your hiring plan will succeed. Instead of giving a single "yes/no" answer, it runs thousands of scenarios with random variation to produce probability distributions.

**Why Monte Carlo?**
- Hiring is inherently uncertain
- Simple formulas can't capture dependencies between roles
- We need confidence intervals, not point estimates

---

## Core Concepts

### 1. Recruiter Capacity

Each recruiter has an `avg_monthly_capacity` — the average number of hires they complete per month.

```
Example: Sarah has capacity 4.5/month
→ She completes ~4-5 hires per month on average
→ Daily rate ≈ 4.5 / 30.44 ≈ 0.148 hires/day
```

### 2. Role Completion Time

The total time to fill a role has two components:

```
Total Time = Queue Wait Time + Process Time
```

| Component | What it represents |
|-----------|-------------------|
| Queue Wait | Time before recruiter starts working on this role |
| Process Time | Time from start to successful hire |

---

## Mathematical Model

### Queue Wait Time

Recruiters work on multiple roles in parallel, but with limited capacity. Roles are prioritized by **urgency score**:

```
urgency_score = days_until_deadline - avg_days_to_hire
```

Lower scores = more urgent = worked on first.

**Wait time formula:**

```
expected_wait_days = (queue_position / monthly_capacity) × 30.44
```

Where:
- `queue_position` = 0, 1, 2, ... (0 = highest priority)
- `monthly_capacity` = recruiter's avg hires/month
- `30.44` = average days per month

**Example:**
```
Recruiter: 3 hires/month capacity
Role position: 4th in queue (position = 3)

expected_wait = (3 / 3) × 30.44 = 30.44 days
```

We model wait time using a **Gamma distribution** to add realistic variance:
- Shape (k) = 2.0 (moderate variance)
- Scale (θ) = expected_wait / 2.0

### Process Time (Gamma Distribution)

Once work begins, the actual hiring time varies around `avg_days_to_hire`. We use the **Gamma distribution** because:

1. It's always positive (can't have negative days)
2. It's right-skewed (some hires take much longer than average)
3. Shape parameter controls predictability

**Gamma Distribution Parameters:**

| Complexity | Shape (k) | Interpretation |
|------------|-----------|----------------|
| Low | 4.0 | More predictable, clustered around mean |
| Medium | 2.5 | Moderate variance |
| High | 1.5 | Less predictable, longer tail |

**Scale parameter:**
```
θ = avg_days_to_hire / k
```

This ensures the mean stays at `avg_days_to_hire` regardless of shape.

**Visual intuition:**

```
Low complexity (k=4):     ▁▃▇██▇▃▁
Medium complexity (k=2.5): ▂▅██▆▃▂▁
High complexity (k=1.5):   ▇█▆▄▃▂▁▁▁  (longer tail)
```

---

## Gamma Distribution Deep Dive

The Gamma distribution is defined by:
- **Shape (k)**: Controls the "peakedness" and skew
- **Scale (θ)**: Stretches/compresses the distribution

**Key properties:**
```
Mean = k × θ
Variance = k × θ²
```

**Why we use it:**
```
If k = 4.0 and avg_days = 40:
  θ = 40 / 4.0 = 10
  Mean = 4.0 × 10 = 40 days ✓
  Std Dev = √(4.0 × 100) = 20 days

If k = 1.5 and avg_days = 40:
  θ = 40 / 1.5 = 26.67
  Mean = 1.5 × 26.67 = 40 days ✓
  Std Dev = √(1.5 × 711) = 32.7 days (more variance!)
```

---

## Monte Carlo Simulation

We run **5,000 simulations** of the entire hiring plan:

```
for simulation 1 to 5000:
    for each role:
        1. Sample queue_wait from Gamma(2.0, expected_wait/2.0)
        2. Sample process_time from Gamma(k_complexity, avg_days/k)
        3. total_time = queue_wait + process_time
        4. Record if total_time > days_until_deadline (missed)
    
    Count total missed deadlines for this simulation
```

### Output Metrics

From 5,000 simulations, we compute:

| Metric | Formula |
|--------|---------|
| Success Rate | `mean(on_time_count) / total_roles` |
| P(role on time) | `sum(completion < deadline) / 5000` |
| Critical Failure Risk | `sum(missed > 5) / 5000` |
| Expected Missed | `mean(missed_per_simulation)` |
| 95% CI | `[percentile(2.5), percentile(97.5)]` |
| P99 Missed | `percentile(99)` of missed counts |

---

## Risk Classification

| Level | Success Probability | Action |
|-------|---------------------|--------|
| 🟢 On Track | ≥ 75% | Monitor normally |
| 🟡 At Risk | 50% - 75% | Needs attention |
| 🔴 High Risk | < 50% | Immediate action required |

---

## Bottleneck Detection

A recruiter is flagged as a **bottleneck** when:

```
failure_ratio = recruiter_failure_rate / overall_avg_failure_rate
```

If `failure_ratio > 1.5`, the recruiter has significantly more failures than average.

---

## Assumptions & Limitations

### Assumptions

1. **Independent roles**: One role's hiring doesn't affect another's timing
2. **Stable capacity**: Recruiter capacity doesn't change over time
3. **Parallel work**: Recruiters work on all assigned roles simultaneously (with divided attention)
4. **Known averages**: `avg_days_to_hire` reflects historical reality

### Limitations

1. **No seasonal effects**: Doesn't account for hiring freezes, holidays
2. **No candidate pool modeling**: Assumes candidates are available
3. **No recruiter skill matching**: Doesn't weight complexity vs. recruiter expertise
4. **Fixed prioritization**: Uses urgency only, not strategic importance

---

## Configuration

Edit `src/config.py` to adjust:

```python
SIMULATION_RUNS = 5000      # More runs = more precision, slower
SIMULATION_MONTHS = 6       # Planning horizon

COMPLEXITY_SHAPE_K = {
    "Low": 4.0,             # Increase for less variance
    "Medium": 2.5,
    "High": 1.5             # Decrease for more variance
}
```

---

## References

- [Gamma Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Gamma_distribution)
- [Monte Carlo Method (Wikipedia)](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [Queueing Theory Basics](https://en.wikipedia.org/wiki/Queueing_theory)
