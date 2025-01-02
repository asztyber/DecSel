import metis
import numpy as np
import pandas as pd
from gdec import graph_creation
from bilp import bilp_test_selection

def split_equations(g, n, **kwargs):
    (edgecuts, parts) = metis.part_graph(g, n, objtype='cut', **kwargs)
    return edgecuts, parts

def eq_assignment_to_subsystems(eq_assignment):
    subsystems = dict()
    for key, value in sorted(eq_assignment.items()):
        subsystems.setdefault(value, []).append(key)
    return subsystems

def subsystems_to_eq_assignment(subsystems):
    return {e: key for key, value in sorted(subsystems.items()) for e in value}

def parts_from_eq_assignment(eq_assignment):
    return [eq_assignment[e] for e in sorted(eq_assignment, key=lambda x: int(x[1:]))]

def subsystems_from_equations_split(g, parts):
    eq_assignment = dict(zip(g.nodes, parts))
    return eq_assignment_to_subsystems(eq_assignment)

def interconnections_for_fmsos(fmsos, eq_assignment, sm):
    n_eqs = len(sm)
    shared_vars = np.zeros((n_eqs, n_eqs))
    for i in range(n_eqs):
        for j in range(n_eqs):
            ei = 'e' + str(i)
            ej = 'e' + str(j)
            if eq_assignment[ei] != eq_assignment[ej]:
                vars1 = sm[ei]
                vars2 = sm[ej]
                shared_vars[i][j] = len(set(vars1).intersection(set(vars2)))
    n_inter = []
    for fmso in fmsos:
        int_fmso = [int(eq[1:]) for eq in fmso]
        n_interconnections = 0
        for i in range(len(int_fmso)):
            for j in range(i):
                n_interconnections += shared_vars[int_fmso[i], int_fmso[j]]
        n_inter.append(n_interconnections)
    return n_inter

def decompose_from_fmsos(fmsos, sm, data, n_subsystems, constraints, small_weights=None, **kwargs):
    g = graph_creation.graph_from_fmsos(fmsos, sm, data, multi_connections=True, constraints=constraints,
                                      small_weights=small_weights)
    edgecuts, parts = split_equations(g, n_subsystems, **kwargs)
    return subsystems_from_equations_split(g, parts), edgecuts

def fmsos_for_subsystems(subsystems, sm, fmsos_eqs, tests, n_faults, min_interconn_est_q=None):
    print('eq eq_assignment')
    eq_assignment = subsystems_to_eq_assignment(subsystems)
    print('count_connections')
    inter_connections = interconnections_for_fmsos(fmsos_eqs, eq_assignment, sm)

    print('df with tests: ')
    df = pd.DataFrame({'tests': tests, 'inter_connections': inter_connections, 'eqs': fmsos_eqs})
    df = df.sort_values(by='inter_connections')
    df = df.drop_duplicates(subset='tests')
    if min_interconn_est_q:
        min_interconn_est = df['inter_connections'].quantile(min_interconn_est_q)
    else:
        min_interconn_est = None

    test_ids = bilp_test_selection.run_search(df, n_faults)

    selected = df.loc[list(test_ids)]
    n_interconnections = selected['inter_connections'].sum()
    fmsos = list(selected['eqs'])
    tests = list(selected['tests'])

    return fmsos, n_interconnections, tests

def loop(initial_subsystems, initial_cost, sm, data, fmsos_eqs, tests, n_faults, n_subsystems, constraints=None,
         min_interconn_est_q=None, max_iter=10, small_weights=None, **kwargs):
    cost = None
    subsystems = initial_subsystems
    steps = []
    costs = [initial_cost]

    best_cost = initial_cost
    best_fmsos = None
    best_tests = None
    best_subsystems = initial_subsystems
    iter_counter = 0

    steps.append(initial_subsystems)
    steps.append(initial_cost)
    while iter_counter < max_iter:
        print("calculating FMSOs")
        fmsos, current_cost, sel_tests = fmsos_for_subsystems(subsystems, sm, fmsos_eqs, tests, n_faults,
                                                            min_interconn_est_q)
        print(current_cost)
        steps.append(fmsos)
        steps.append(current_cost)
        if cost is not None and cost == current_cost:
            break
        cost = current_cost

        if not best_cost or current_cost < best_cost:
            best_cost = current_cost
            best_fmsos = fmsos
            best_tests = sel_tests
            best_subsystems = subsystems

        costs.append(best_cost)

        print("calculating subsystems")
        subsystems, n_interconnections = decompose_from_fmsos(fmsos, sm, data, n_subsystems, constraints,
                                                          small_weights=small_weights, **kwargs)
        eq_assignment = subsystems_to_eq_assignment(subsystems)
        current_cost = sum([count_fmso_interconnections(fmso, eq_assignment, sm) for fmso in fmsos])
        print(current_cost, n_interconnections)
        steps.append(subsystems)
        steps.append(current_cost)

        if cost == current_cost:
            break
        cost = current_cost

        if current_cost < best_cost:
            best_cost = current_cost
            best_subsystems = subsystems
            best_fmsos = fmsos
            best_tests = sel_tests

        costs.append(best_cost)
        iter_counter += 1

    return best_fmsos, best_subsystems, best_tests, steps, costs

def max_isolability(tests, n_faults):
    tests = list(set(tests))
    test_ids = range(len(tests))

    per_fault = dict()
    for fault in range(n_faults):
        fault_tests = set()
        for test_id, test in zip(test_ids, tests):
            if fault in test:
                fault_tests.add(test_id)
        per_fault[fault] = tuple(sorted(fault_tests))

    isolability = dict()
    for key, value in sorted(per_fault.items()):
        isolability.setdefault(value, []).append(key)

    return list(isolability.values())

def count_fmso_interconnections(fmso, eq_assignment, sm):
    n_interconnections = 0
    for e1 in fmso:
        for e2 in fmso:
            if eq_assignment[e1] != eq_assignment[e2]:
                vars1 = sm[e1]
                vars2 = sm[e2]
                n_interconnections = n_interconnections + len(set(vars1).intersection(set(vars2)))
    return n_interconnections / 2

def select_fmsos_without_decomposition(fmsos_eqs, tests, n_faults):
    """
    Selects a minimal set of FMSOs without considering decomposition costs.
    
    Args:
        fmsos_eqs: List of FMSOs (each FMSO is a list of equations)
        tests: List of test vectors (each test vector shows which faults are detected)
        n_faults: Number of faults in the system
        
    Returns:
        tuple: (selected_fmsos, selected_tests) - Lists of selected FMSOs and their corresponding tests
    """
    # Create DataFrame with tests and equations
    df = pd.DataFrame({'tests': tests, 'eqs': fmsos_eqs})
    df = df.drop_duplicates(subset='tests')  # Remove duplicate tests
    
    # Use existing BILP selection since it already minimizes the number of tests
    test_ids = bilp_test_selection.run_search(df, n_faults)
    
    # Get selected FMSOs and tests
    selected = df.loc[list(test_ids)]
    selected_fmsos = list(selected['eqs'])
    selected_tests = list(selected['tests'])
    
    return selected_fmsos, selected_tests
