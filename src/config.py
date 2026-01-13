"""Configuration constants for the Hiring Feasibility Engine."""

from datetime import date

TODAY = date.today()
SIMULATION_RUNS = 5000
SIMULATION_MONTHS = 6

COMPLEXITY_SHAPE_K = {
    "Low": 4.0,
    "Medium": 2.5,
    "High": 1.5
}
