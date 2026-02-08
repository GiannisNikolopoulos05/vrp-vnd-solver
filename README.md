# VRP Solver (Warehouse Picking) — Cheapest Insertion + VND (Python)

University project: heuristic solver for a Vehicle Routing Problem (VRP) in warehouse picking with capacity constraints and SKU-family visit requirements.

## Approach
- Construction: **Cheapest Insertion**
- Improvement: **Variable Neighborhood Descent (VND)**
  - relocate / swap / 2-opt
  - same-family node replacement
- Feasibility: capacity & required visits validation
- Evaluation: solution cost validated with provided checker (not included here)

## How to run
```bash
python our_solution.py <instance_file> <seed> <time_limit>

Note: Course instances and checker are not included in this repository.
