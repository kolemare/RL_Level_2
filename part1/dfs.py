import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_graph(num_vertices, num_edges):
    G = nx.Graph()
    G.add_nodes_from(range(1, num_vertices + 1))
    while len(G.edges) < num_edges:
        v1 = random.randint(1, num_vertices)
        v2 = random.randint(1, num_vertices)
        if v1 != v2:
            G.add_edge(v1, v2)
    return G

def dfs(graph, start_vertex):
    visited = set()
    order = []
    stack = [start_vertex]

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            # Add vertices in reverse order so that the first vertex is processed first
            stack.extend(reversed(sorted(graph[vertex])))

    return order

def visualize_graph(graph, path=None):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=1)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=2)

    plt.axis('off')
    plt.show()

# Graph parameters
num_vertices = 10
num_edges = 15

# Generate the graph
G = generate_graph(num_vertices, num_edges)
print("Generated Graph:", G.edges())

# Perform DFS
start_vertex = 1
dfs_order = dfs(G, start_vertex)
print("DFS Order starting from vertex", start_vertex, ":", dfs_order)

# Visualize the graph with DFS path highlighted
visualize_graph(G, dfs_order)