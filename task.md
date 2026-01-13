# The "Hiring Feasibility" Engine

## Context

You are a Senior Engineer at _Acme Inc._ Our Head of Talent has just finalized the Q1/Q2 Hiring Plan. However, they are nervous. In previous quarters, we missed critical hiring deadlines because we didn't accurately account for how "noisy" the recruiting process is.

Management wants a tool to scientifically validate if our current plan is realistic given our team size. They are tired of "gut feelings", they want data.

## The Objective

Build a **Capacity Planning Dashboard** (CLI or Web UI) that ingests our list of recruiters and our hiring plan, then outputs a **global confidence assessment** for the entire plan.

**Assumption:** The plan is active as of _Today_ (use the current system date where applicable).

## The Data

You have been provided with two JSON files:

1. **`recruiters.json`**:
* The list of 9 recruiters.
* `avg_monthly_capacity`: The historical average number of hires this recruiter completes per month.

2. **`hiring_plan.json`**:
* A list of 60 open roles we need to fill.
* `target_start_date`: When the employee needs to start.
* `avg_days_to_hire`: Historical average time to fill this type of role.
* `complexity`: High/Medium/Low (indicates the volatility/difficulty of the search).
* `assigned_recruiter_id`: The specific recruiter already assigned to work on this role.


## Core Requirements

1. **Data Ingestion:**
Load and parse the provided JSON files.

2. **Modeling Logic:**
* Determine when each role is likely to be filled based on the assigned recruiter's capacity.
* *Note:* Recruiters often have multiple roles assigned to them. You must account for how their capacity is distributed or queued.

3. **Global Plan Feasibility:**
* Since target dates are fixed, Management needs to know the Probability of Success.
* Instead of a simple "Yes/No", calculate:
  * Overall Success Rate: "We project that only [X]% of roles will start on time."
  * Critical Failure Risk: "There is a [Y]% probability that we miss more than 5 deadlines."

4. **Bonus: Bottleneck Detection:**
* Your tool should be able to identify the "weak links."
* Which specific recruiter assignments or role complexities are the biggest risks?
* *Example Output:* "Recruiter [Name] is overloaded; swapping [Role ID] would improve the plan's success probability by 15%."

5. **Presentation:**
* Display the results clearly.
* You can build a CLI table, a simple HTML report, or a dashboard using a any framework.

## Constraints & Rules

* **Time:** You have **90 minutes** to build a working prototype.
* **Tools:** You are **encouraged** to use LLMs to generate code, write boilerplate, or design the UI.
* **Libraries:** You may use any Python/JS libraries you wish.

## Evaluation Criteria

We are not just looking for code that runs. We are looking for:

* **Engineering Judgment:** How do you handle ambiguity in the data?
* **Modeling Strategy:** How do you translate "business requirements" into "mathematical logic"?
* **LLM Usage:** Can you effectively direct an AI to build a complex tool quickly?
* **Completeness**: While we don't expect you to complete the whole thing in 90 mins, we will evaluate how far you have come in the limited amount of time

## Hint

Don't just subtract hours. If a recruiter has an average capacity of 1.0 hire/month, that doesn't mean they hire exactly one person every 30 days like a robot. Real life has randomness. Build a tool that tells us how *lucky* we need to be to hit these dates. You may try Googling for _Poisson Distribution_.
