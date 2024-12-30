import networkx as nx


def graph_from_fmsos(fmsos, sm, data, multi_connections=False, constraints=None, small_weights=None):
    n_eq = len(sm.keys())
    if constraints is not None:
        g = create_constraints_graph(constraints, data)
    else:
        g = nx.Graph()
    g.add_nodes_from(['e' + str(i) for i in range(n_eq)])
    g.graph['edge_weight_attr'] = 'weight'

    if small_weights:
        g.graph['node_weight_attr'] = 'weight'
        for node in g.nodes:
            if 'weight' in g.nodes[node]:
                g.nodes[node]['weight'] = g.nodes[node]['weight'] * 1000
            elif node in small_weights:
                g.nodes[node]['weight'] = 1
            else:
                g.nodes[node]['weight'] = 1000

    for fmso in fmsos:
        for e1 in fmso:
            for e2 in fmso:
                if e1 != e2:
                    vars1 = sm[e1]
                    vars2 = sm[e2]
                    weight = len(set(vars1).intersection(set(vars2)))
                    if weight > 0:
                        if not g.has_edge(e1, e2):
                            g.add_edge(e1, e2, weight=weight)
                        else:
                            if multi_connections:
                                g[e1][e2]['weight'] = g[e1][e2]['weight'] + weight

    if multi_connections:
        for e1, e2 in g.edges:
            g[e1][e2]['weight'] = int(g[e1][e2]['weight']/2)

    return g


def create_constraints_graph(constraints, data):
    big_edge_weight = len(data['unknown']) * len(data['model'])
    big_node_weight = len(data['model'])

    g = nx.Graph()
    for c in constraints:
        for e1 in c:
            g.add_node(e1, weight=int(big_node_weight / len(c)))
            for e2 in c:
                if e1 != e2 and not g.has_edge(e1, e2):
                    g.add_edge(e1, e2, weight=big_edge_weight)

    g.graph['edge_weight_attr'] = 'weight'
    g.graph['node_weight_attr'] = 'weight'

    return g
