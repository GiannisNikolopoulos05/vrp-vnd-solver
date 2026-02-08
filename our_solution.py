import sys
import time
import random


# =========================================================
# MODEL
# Διαβάζω το instance και κρατάω:
# - capacity, αριθμό οχημάτων
# - demand ανά οικογένεια
# - cost matrix (κόστος μετακίνησης i -> j)
# Επίσης φτιάχνω βοηθητικές δομές ώστε να ξέρω:
# - σε ποια οικογένεια ανήκει κάθε κόμβος (family_of)
# - ποιο demand έχει κάθε κόμβος (demand_of)
# - ποιοι κόμβοι ανήκουν σε κάθε οικογένεια (nodes_by_family)
# =========================================================
class Model:
    def __init__(self):
        self.num_nodes = 0
        self.num_fam = 0
        self.num_req = 0
        self.capacity = 0
        self.vehicles = 0

        self.fam_members = []
        self.fam_req = []
        self.fam_demand = []

        self.cost = []
        self.family_of = []
        self.demand_of = []
        self.nodes_by_family = []

    def BuildModel(self, instance_path):
        # Διαβάζω τις 4 πρώτες γραμμές όπως ορίζει η εκφώνηση
        with open(instance_path, 'r') as f:
            a = f.readline().split()
            self.num_nodes = int(a[0])
            self.num_fam = int(a[1])
            self.num_req = int(a[2])
            self.capacity = int(a[3])
            self.vehicles = int(a[4])

            self.fam_members = list(map(int, f.readline().split()))
            self.fam_req = list(map(int, f.readline().split()))
            self.fam_demand = list(map(int, f.readline().split()))

            # Διαβάζω τον πίνακα κόστους.
            # Κάνω -1 -> 10000 ακριβώς όπως κάνει ο sol_checker,
            # ώστε ο υπολογισμός του Cost να συμφωνεί 100%.
            self.cost = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [int(x) for x in line.split()]
                row = [10000 if x < 0 else x for x in row]
                self.cost.append(row)

        # Φτιάχνω βοηθητικούς πίνακες για γρήγορη πρόσβαση
        self._BuildHelpers()

    def _BuildHelpers(self):
        # family_of[node] = id οικογένειας του node (για depot βάζω None)
        self.family_of = [None] * (self.num_nodes + 1)
        self.family_of[0] = None

        # Σύμφωνα με το instance: οι κόμβοι είναι ομαδοποιημένοι ανά οικογένεια
        idx = 1
        for fam_id, cnt in enumerate(self.fam_members):
            for _ in range(cnt):
                if idx <= self.num_nodes:
                    self.family_of[idx] = fam_id
                    idx += 1

        # demand_of[node] = demand της οικογένειας του node (depot demand = 0)
        self.demand_of = [0] * (self.num_nodes + 1)
        for node in range(1, self.num_nodes + 1):
            fam = self.family_of[node]
            self.demand_of[node] = self.fam_demand[fam]

        # nodes_by_family[f] = λίστα κόμβων που ανήκουν στην οικογένεια f
        self.nodes_by_family = [[] for _ in range(self.num_fam)]
        for node in range(1, self.num_nodes + 1):
            self.nodes_by_family[self.family_of[node]].append(node)


# =========================================================
# COST FUNCTIONS
# Από τη θεωρία VRP: κόστος λύσης = άθροισμα ακμών των routes.
# (δηλαδή αθροίζω cost[i][j] για κάθε διαδοχικό ζευγάρι i -> j)
# =========================================================
def RouteCost(route, cost):
    c = 0
    for i in range(len(route) - 1):
        c += cost[route[i]][route[i + 1]]
    return c


def SolutionCost(routes, cost):
    total = 0
    for r in routes:
        total += RouteCost(r, cost)
    return total


def ComputeLoads(m, routes):
    # Χρησιμοποιώ το demand_of που έφτιαξα στο Model
    loads = []
    for r in routes:
        loads.append(sum(m.demand_of[x] for x in r))
    return loads


# =========================================================
# CONSTRUCTION PHASE
# 1) Επιλέγω ακριβώς fam_req[f] κόμβους από κάθε οικογένεια f.
# 2) Τους βάζω σε routes με Cheapest Insertion, τηρώντας capacity.
#
# Αν χρησιμοποιώ τυχαιότητα, το κάνω μέσω rnd που είναι seeded με seed ∈ {1..5}.
# =========================================================
def SelectCustomers(m, rnd):
    selected = []
    for fam_id in range(m.num_fam):
        req = m.fam_req[fam_id]
        cand = m.nodes_by_family[fam_id][:]

        # Βασικό “φτηνό” κριτήριο: κόστος depot->node + node->depot
        cand.sort(key=lambda x: m.cost[0][x] + m.cost[x][0])

        # Για να μην είναι πάντα ίδια επιλογή, κάνω μικρή τυχαιότητα σε top pool
        pool_size = min(len(cand), max(req * 2, req))
        pool = cand[:pool_size]
        rnd.shuffle(pool)
        pool.sort(key=lambda x: m.cost[0][x] + m.cost[x][0])

        selected.extend(pool[:req])

    return selected


def CheapestInsertionConstruction(m, selected, rnd):
    # Αρχικά κάθε όχημα έχει διαδρομή 0-0
    routes = [[0, 0] for _ in range(m.vehicles)]
    loads = [0] * m.vehicles

    # Ανακατεύω τη σειρά εισαγωγής (controlled από seed)
    selected = selected[:]
    rnd.shuffle(selected)

    # Για κάθε πελάτη/κόμβο, βρίσκω την εισαγωγή με το μικρότερο “delta cost”
    for cust in selected:
        d = m.demand_of[cust]
        best = None  # (delta, r_idx, pos)

        for r_idx in range(m.vehicles):
            if loads[r_idx] + d > m.capacity:
                continue

            r = routes[r_idx]
            for pos in range(len(r) - 1):
                a = r[pos]
                b = r[pos + 1]
                delta = m.cost[a][cust] + m.cost[cust][b] - m.cost[a][b]
                if best is None or delta < best[0]:
                    best = (delta, r_idx, pos + 1)

        # Αν δεν βρεθεί θέση, η κατασκευή απέτυχε (σπάνιο αν instance feasible)
        if best is None:
            return None

        _, r_idx, ins_pos = best
        routes[r_idx].insert(ins_pos, cust)
        loads[r_idx] += d

    return routes


# =========================================================
# MOVE STRUCTURES
# Κρατάω τις κινήσεις σε αντικείμενα για να είναι “καθαρή” η αναζήτηση:
# - Relocation: μεταφέρω έναν κόμβο σε άλλο route/θέση
# - Swap: ανταλλαγή δύο κόμβων (ίδιο ή διαφορετικό route)
# - 2-opt: αναστροφή τμήματος σε route
# - FamilyReplace: αντικαθιστώ κόμβο με άλλον της ίδιας οικογένειας
#   (ίδιο demand, άρα κρατάω feasibility για family requirements)
# =========================================================
class RelocationMove:
    def __init__(self):
        self.originRoute = None
        self.targetRoute = None
        self.originIndex = None
        self.targetIndex = None
        self.moveCost = 10**18

    def Initialize(self):
        self.originRoute = None
        self.targetRoute = None
        self.originIndex = None
        self.targetIndex = None
        self.moveCost = 10**18


class SwapMove:
    def __init__(self):
        self.route1 = None
        self.route2 = None
        self.index1 = None
        self.index2 = None
        self.moveCost = 10**18

    def Initialize(self):
        self.route1 = None
        self.route2 = None
        self.index1 = None
        self.index2 = None
        self.moveCost = 10**18


class TwoOptMove:
    def __init__(self):
        self.route = None
        self.i = None
        self.j = None
        self.moveCost = 10**18

    def Initialize(self):
        self.route = None
        self.i = None
        self.j = None
        self.moveCost = 10**18


class FamilyReplaceMove:
    def __init__(self):
        self.route = None
        self.index = None
        self.oldNode = None
        self.newNode = None
        self.moveCost = 10**18

    def Initialize(self):
        self.route = None
        self.index = None
        self.oldNode = None
        self.newNode = None
        self.moveCost = 10**18


# =========================================================
# SOLVER (VND)
# VND = Variable Neighborhood Descent:
# περνάω διαδοχικά από γειτονιές (relocate, swap, 2-opt, family replace).
# Όταν βρω βελτίωση (moveCost < 0), εφαρμόζω την κίνηση και γυρνάω στο k=0.
# Αυτό είναι ακριβώς η λογική VND από την ύλη.
# =========================================================
class Solver:
    def __init__(self, m):
        self.m = m
        self.routes = None
        self.loads = None
        self.used = None  # ποιοι κόμβοι είναι ήδη στη λύση

    def BuildInitialSolution(self, rnd):
        # Construction: πρώτα διαλέγω κόμβους, μετά τους τοποθετώ σε routes
        selected = SelectCustomers(self.m, rnd)
        routes = CheapestInsertionConstruction(self.m, selected, rnd)
        if routes is None:
            return None

        self.routes = routes
        self.loads = ComputeLoads(self.m, self.routes)

        # used set για να ξέρω τι έχω ήδη επιλέξει (χρήσιμο στο FamilyReplace)
        self.used = set()
        for r in self.routes:
            for k in range(1, len(r) - 1):
                self.used.add(r[k])

        return self.routes

    # ---------- Relocation ----------
    def FindBestRelocation(self, rm):
        m = self.m
        cost = m.cost
        rm.Initialize()

        for r1_idx in range(len(self.routes)):
            r1 = self.routes[r1_idx]
            if len(r1) <= 2:
                continue

            for i in range(1, len(r1) - 1):
                B = r1[i]
                d = m.demand_of[B]

                A = r1[i - 1]
                C = r1[i + 1]
                # κόστος αφαίρεσης B από r1
                remove_delta = cost[A][C] - cost[A][B] - cost[B][C]

                for r2_idx in range(len(self.routes)):
                    if r2_idx == r1_idx:
                        continue
                    if self.loads[r2_idx] + d > m.capacity:
                        continue

                    r2 = self.routes[r2_idx]
                    for j in range(0, len(r2) - 1):
                        F = r2[j]
                        G = r2[j + 1]
                        # κόστος εισαγωγής B ανάμεσα σε F και G
                        insert_delta = cost[F][B] + cost[B][G] - cost[F][G]

                        move_cost = remove_delta + insert_delta
                        if move_cost < rm.moveCost:
                            rm.moveCost = move_cost
                            rm.originRoute = r1_idx
                            rm.targetRoute = r2_idx
                            rm.originIndex = i
                            rm.targetIndex = j

    def ApplyRelocation(self, rm):
        node = self.routes[rm.originRoute].pop(rm.originIndex)
        d = self.m.demand_of[node]
        self.loads[rm.originRoute] -= d

        self.routes[rm.targetRoute].insert(rm.targetIndex + 1, node)
        self.loads[rm.targetRoute] += d

    # ---------- Swap ----------
    def FindBestSwap(self, sm):
        m = self.m
        cost = m.cost
        sm.Initialize()

        for r1_idx in range(len(self.routes)):
            r1 = self.routes[r1_idx]
            if len(r1) <= 2:
                continue

            for r2_idx in range(r1_idx, len(self.routes)):
                r2 = self.routes[r2_idx]
                if len(r2) <= 2:
                    continue

                for i in range(1, len(r1) - 1):
                    n1 = r1[i]
                    d1 = m.demand_of[n1]
                    a1 = r1[i - 1]
                    b1 = r1[i + 1]

                    for j in range(1, len(r2) - 1):
                        if r1_idx == r2_idx and i == j:
                            continue
                        # απλοποίηση: δεν κάνω swap σε διπλανά σημεία στο ίδιο route
                        if r1_idx == r2_idx and (i + 1 == j or j + 1 == i):
                            continue

                        n2 = r2[j]
                        d2 = m.demand_of[n2]
                        a2 = r2[j - 1]
                        b2 = r2[j + 1]

                        # capacity feasibility αν είναι διαφορετικά routes
                        if r1_idx != r2_idx:
                            if self.loads[r1_idx] - d1 + d2 > m.capacity:
                                continue
                            if self.loads[r2_idx] - d2 + d1 > m.capacity:
                                continue

                        before = cost[a1][n1] + cost[n1][b1] + cost[a2][n2] + cost[n2][b2]
                        after = cost[a1][n2] + cost[n2][b1] + cost[a2][n1] + cost[n1][b2]
                        delta = after - before

                        if delta < sm.moveCost:
                            sm.moveCost = delta
                            sm.route1 = r1_idx
                            sm.route2 = r2_idx
                            sm.index1 = i
                            sm.index2 = j

    def ApplySwap(self, sm):
        r1 = self.routes[sm.route1]
        r2 = self.routes[sm.route2]

        n1 = r1[sm.index1]
        n2 = r2[sm.index2]

        r1[sm.index1], r2[sm.index2] = n2, n1

        if sm.route1 != sm.route2:
            d1 = self.m.demand_of[n1]
            d2 = self.m.demand_of[n2]
            self.loads[sm.route1] = self.loads[sm.route1] - d1 + d2
            self.loads[sm.route2] = self.loads[sm.route2] - d2 + d1

    # ---------- Two-Opt ----------
    def FindBestTwoOpt(self, tm):
        m = self.m
        cost = m.cost
        tm.Initialize()

        for r_idx in range(len(self.routes)):
            r = self.routes[r_idx]
            if len(r) <= 4:
                continue

            for i in range(1, len(r) - 2):
                for j in range(i + 1, len(r) - 1):
                    A = r[i - 1]
                    B = r[i]
                    C = r[j]
                    D = r[j + 1]

                    before = cost[A][B] + cost[C][D]
                    after = cost[A][C] + cost[B][D]
                    delta = after - before

                    if delta < tm.moveCost:
                        tm.moveCost = delta
                        tm.route = r_idx
                        tm.i = i
                        tm.j = j

    def ApplyTwoOpt(self, tm):
        r = self.routes[tm.route]
        i = tm.i
        j = tm.j
        # Αναστροφή τμήματος [i..j]
        r[i:j + 1] = reversed(r[i:j + 1])

    # ---------- Family Replace ----------
    def FindBestFamilyReplace(self, fm):
        m = self.m
        cost = m.cost
        fm.Initialize()

        for r_idx in range(len(self.routes)):
            r = self.routes[r_idx]
            for i in range(1, len(r) - 1):
                oldNode = r[i]
                fam = m.family_of[oldNode]
                A = r[i - 1]
                C = r[i + 1]

                # Δοκιμάζω άλλο κόμβο της ίδιας οικογένειας που δεν έχει επιλεγεί ήδη
                # (ίδιο demand => δεν αλλάζει feasibility των family requirements)
                for cand in m.nodes_by_family[fam]:
                    if cand == oldNode:
                        continue
                    if cand in self.used:
                        continue

                    delta = (cost[A][cand] + cost[cand][C]) - (cost[A][oldNode] + cost[oldNode][C])
                    if delta < fm.moveCost:
                        fm.moveCost = delta
                        fm.route = r_idx
                        fm.index = i
                        fm.oldNode = oldNode
                        fm.newNode = cand

    def ApplyFamilyReplace(self, fm):
        r = self.routes[fm.route]
        oldNode = fm.oldNode
        newNode = fm.newNode

        r[fm.index] = newNode
        self.used.remove(oldNode)
        self.used.add(newNode)

    # ---------- VND loop ----------
    def VND(self, endTime):
        rm = RelocationMove()
        sm = SwapMove()
        tm = TwoOptMove()
        fm = FamilyReplaceMove()

        # k=0..3: περνάω από διαφορετικές γειτονιές
        k = 0
        while k < 4 and time.time() < endTime:
            if k == 0:
                self.FindBestRelocation(rm)
                if rm.originRoute is not None and rm.moveCost < 0:
                    self.ApplyRelocation(rm)
                    k = 0  # βελτίωση -> ξαναξεκινάω από την 1η γειτονιά
                else:
                    k += 1

            elif k == 1:
                self.FindBestSwap(sm)
                if sm.route1 is not None and sm.moveCost < 0:
                    self.ApplySwap(sm)
                    k = 0
                else:
                    k += 1

            elif k == 2:
                self.FindBestTwoOpt(tm)
                if tm.route is not None and tm.moveCost < 0:
                    self.ApplyTwoOpt(tm)
                    k = 0
                else:
                    k += 1

            else:
                self.FindBestFamilyReplace(fm)
                if fm.route is not None and fm.moveCost < 0:
                    self.ApplyFamilyReplace(fm)
                    k = 0
                else:
                    k += 1


# =========================================================
# OUTPUT + CHECKER
# Γράφω our_solution.txt ακριβώς στο format του sample.
# Μετά (προαιρετικά) τρέχω sol_checker.py για επιβεβαίωση.
# =========================================================
def WriteSolutionFile(filename, routes, cost_value):
    with open(filename, 'w') as f:
        f.write("Cost: " + str(int(cost_value)) + "\n")
        for r in routes:
            f.write("-".join(str(x) for x in r) + "\n")


def TryRunChecker(instance_path, sol_path):
    try:
        import runpy
        chk = runpy.run_path("sol_checker.py")
        model = chk["create_model"](instance_path)
        sol = chk["load_solution"](model, sol_path)
        ok = chk["check_feasibility"](model, sol)
        if ok:
            print("Checker: Valid solution. Cost:", sol.cost)
        else:
            print("Checker: Invalid solution.")
    except Exception as e:
        print("Checker did not run:", e)


# =========================================================
# MAIN
# - παίρνω seed μόνο από {1,2,3,4,5} (όπως ζητάει το PDF)
# - τρέχω μέχρι ~4 λεπτά max (default 240 sec), αλλά μπορεί να σταματήσει νωρίτερα
#   αν δεν βρίσκει βελτίωση για αρκετή ώρα.
# =========================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python our_solution.py instance.txt seed(1..5) [timeLimitSeconds]")
        return

    instance_path = sys.argv[1]
    seed = int(sys.argv[2])

    if seed not in [1, 2, 3, 4, 5]:
        print("Seed must be one of: 1,2,3,4,5")
        return

    time_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 240

    m = Model()
    m.BuildModel(instance_path)

    start = time.time()
    endTime = start + time_limit

    # Ένα RNG seeded μόνο με 1..5. Δεν κάνω reseed μέσα στο loop.
    rnd = random.Random(seed)

    best_routes = None
    best_cost = 10**18

    # Early stop: αν δεν υπάρχει καμία βελτίωση για Χ sec, σταματάω νωρίτερα
    no_improve_limit = 35
    last_improve_time = time.time()

    attempt = 0
    while time.time() < endTime:
        attempt += 1

        solver = Solver(m)
        routes = solver.BuildInitialSolution(rnd)
        if routes is None:
            continue

        # VND βελτιώνει τη λύση μέχρι να “κολλήσει” ή μέχρι να τελειώσει ο χρόνος
        solver.VND(endTime)

        c = SolutionCost(solver.routes, m.cost)
        if c < best_cost:
            best_cost = c
            best_routes = [r[:] for r in solver.routes]
            last_improve_time = time.time()
            print("New best:", int(best_cost), "| attempt:", attempt)

        if best_routes is not None and (time.time() - last_improve_time > no_improve_limit):
            break

    out_file = "our_solution.txt"
    WriteSolutionFile(out_file, best_routes, best_cost)
    print("Wrote:", out_file, "| Cost:", int(best_cost))

    TryRunChecker(instance_path, out_file)


if __name__ == "__main__":
    main()
