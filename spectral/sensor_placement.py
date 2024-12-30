import networkx as nx
from sklearn.cluster import SpectralClustering
import pandas as pd


def place_sensors(wn, n_sensors):
    diameter = wn.query_link_attribute('diameter')
    G = wn.get_graph(wn, link_weight=diameter)
    G = nx.Graph(G)

    adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
    sc = SpectralClustering(n_clusters=n_sensors, affinity='precomputed', random_state=0)
    clustering = sc.fit(adj_matrix)
    node_attribute = pd.Series(data=clustering.labels_, index=wn.node_name_list)
    sensors = []
    for i in range(n_sensors):
        cluster_nodes = list(node_attribute[node_attribute == i].index)
        G_sub = G.subgraph(cluster_nodes)
        if len(cluster_nodes) < 3:
            sensors.append(cluster_nodes[0])
        else:
            centrality = pd.Series(nx.eigenvector_centrality_numpy(G_sub))
            sensors.append(centrality.idxmax())
    return sensors
