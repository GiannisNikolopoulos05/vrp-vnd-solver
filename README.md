# VRP Solver (Warehouse Picking) — Cheapest Insertion + VND (Python)

University project: a heuristic solver for a **Vehicle Routing Problem (VRP)** in a warehouse picking setting, with **capacity constraints** and **SKU-family visit requirements**. The objective is to **minimize routing cost**.

## Overview
This repository contains a Python implementation that:
- builds an initial feasible solution using **Cheapest Insertion**
- improves it using **Variable Neighborhood Descent (VND)** with local search moves
- outputs routes in the required solution format for external validation

## Note
Course/competition instances and the provided solution checker are **not included** in this repository.

---

## Problem Summary (Warehouse Picking VRP)
We solve a warehouse picking **Vehicle Routing Problem (VRP)** where picker vehicles collect ordered items stored across warehouse shelves and return them to a central location (depot). The goal is to **minimize total routing cost**.

**Scenario (high level):**
- E-commerce warehouse with many SKUs stored using **scattered storage / mixed shelves** (same SKU in multiple locations).
- A homogeneous fleet of **13 picker vehicles**, each with **capacity 270**.
- All routes **start and end at the depot (node 0)**.
- Each item belongs to exactly one **SKU family**; items in the same family have equal demand.
- Each family has a **required number of items to pick** (not all nodes must be visited).
- Each customer node can be visited **at most once**, while respecting **vehicle capacity** constraints.

---

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

---

## Repository contents
- `our_solution.py` — main solver (construction + VND improvement)

---

## How to run
Example usage (as implemented in `our_solution.py`):

```bash
python our_solution.py <instance_file> <seed>
