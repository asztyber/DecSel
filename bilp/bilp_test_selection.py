import numpy as np
from gurobipy import Model, GRB, quicksum


def df_to_bdm(df, n_faults):
    print('df to bdm')
    n_tests = len(df)
    bdm = np.zeros((n_tests, n_faults))

    for i, test in enumerate(df['tests']):
        for fault in test:
            bdm[i][fault - 1] = 1

    return bdm


def optimize(bdm, costs):
    print('optimize')
    n_tests, n_faults = bdm.shape
    # column with zeros - for detectability
    bdm = np.c_[bdm, np.zeros(n_tests)]
    n_faults = n_faults + 1  # dummy fault for detectabiblity

    f = np.zeros((n_tests, n_faults, n_faults))
    print('matrix construction: ')
    for i in range(n_tests):
        for j in range(n_faults):
            for k in range(0, j):
                if ((bdm[i][k]) != (bdm[i][j])):
                    f[i][k][j] = 1
                    f[i][j][k] = 1

    print('end of matrix construction: ')
    model = Model('test selection')
    model.setParam("Seed", 0)
    z = model.addVars(n_tests, vtype=GRB.BINARY)

    model.Params.iterationlimit = 1000000
    print('constraints add')
    for j in range(0, n_faults):
        for k in range(0, j):
            model.addConstr(quicksum(f[i][j][k] * z[i] for i in range(n_tests)) >= 1)

    model.setObjective(z.prod(costs), GRB.MINIMIZE)
    print('gurobi start')
    model.optimize()

    selected_tests = []
    for i, v in enumerate(model.getVars()):
        if v.x == 1:
            selected_tests.append(i)

    return selected_tests


def run_search(df, n_faults):
    bdm = df_to_bdm(df, n_faults)
    selected_tests = optimize(bdm, list(df['inter_connections']))

    return df.index[selected_tests]
