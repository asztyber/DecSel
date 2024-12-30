from random import random, randint
import pandas as pd


def generate_random_tests(n_tests, fault_cnt, faults_factor):
    tests = []
    while len(tests) < n_tests:
        test = []
        for x in range(0, fault_cnt):
            if random() < faults_factor:
                test.append(x)
        test = tuple(sorted(test))
        if test and test not in tests:
            tests.append(test)

    return tests


def generate_random_input(n_tests, fault_cnt, faults_factor=0.14, conn_low=2, conn_high=30):
    tests = generate_random_tests(n_tests, fault_cnt, faults_factor)
    interconnections = [randint(conn_low, conn_high) for _ in range(n_tests)]
    df = pd.DataFrame({'tests': tests, 'inter_connections': interconnections})
    df = df.sort_values(by='inter_connections')
    df = df.drop_duplicates(subset='tests')
    return df
