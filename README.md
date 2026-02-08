# VRP Solver (Warehouse Picking) — Cheapest Insertion + VND (Python)

University project: a heuristic solver for a **Vehicle Routing Problem (VRP)** in a warehouse picking setting, with **capacity constraints** and **SKU-family visit requirements**. The objective is to **minimize routing cost**.

## Overview
This repository contains a Python implementation that:
- builds an initial feasible solution using **Cheapest Insertion**
- improves it using **Variable Neighborhood Descent (VND)** with local search moves
- outputs routes in the required solution format for external validation

## Problem (high level)
- A set of customer nodes (warehouse locations) must be served by a fixed number of vehicles.
- Each node belongs to a **family (SKU group)**.
- Vehicles have a **capacity** constraint.
- Only a required number of visits per family is needed (not necessarily all nodes).
- Goal: **minimize total travel cost**.

## Approach
### 1) Construction (Initial Solution)
- **Cheapest Insertion:** iteratively inserts nodes into routes where the marginal cost increase is smallest, while maintaining feasibility (capacity + family requirements).

### 2) Improvement (VND / Local Search)
Applies multiple neighborhoods to reduce total cost:
- **Relocate** (move a node to a different position/route)
- **Swap** (swap two nodes)
- **2-opt** (reverse segments to reduce route cost)
- **Same-family replacement** (replace a visited node with another node of the same family if it improves cost)

Search is typically **seeded** and **time-limited**.

## Repository contents
- `our_solution.py` — main solver (construction + VND improvement)

## How to run
Example usage (as implemented in `our_solution.py`):

```bash
python our_solution.py <instance_file> <seed> <time_limit>

