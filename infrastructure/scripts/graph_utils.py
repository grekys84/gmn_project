import torch
import networkx as nx
import math
from torch_geometric.data import Data

def pyg_to_networkx(data):
    G = nx.Graph()
    for i in range(data.x.shape[0]):
        G.add_node(i, x=data.x[i, 0].item(), y=data.x[i, 1].item())
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        edges = data.edge_index.t().tolist()
        G.add_edges_from([(int(u), int(v)) for u, v in edges])
    return G

def networkx_to_pyg(G):
    node_coords = []
    node_mapping = {}
    for i, (node_id, attrs) in enumerate(G.nodes(data=True)):
        node_mapping[node_id] = i
        node_coords.append([attrs['x'], attrs['y']])
    x = torch.tensor(node_coords, dtype=torch.float) if node_coords else torch.empty((0, 2), dtype=torch.float)

    edges = []
    for u, v in G.edges():
        if u in node_mapping and v in node_mapping:
            edges.append([node_mapping[u], node_mapping[v]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def remove_frame_by_longest_edges_keep_nodes(graph_data, num_edges_to_remove=4):
    if graph_data is None or not hasattr(graph_data, 'x') or graph_data.x is None or graph_data.x.shape[0] == 0:
        return graph_data

    G = pyg_to_networkx(graph_data)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return graph_data

    edge_lengths = []
    for u, v in G.edges():
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        edge_lengths.append((length, u, v))

    edge_lengths.sort(reverse=True)
    edges_to_remove = edge_lengths[:num_edges_to_remove]
    G.remove_edges_from([(u, v) for _, u, v in edges_to_remove])

    return networkx_to_pyg(G)