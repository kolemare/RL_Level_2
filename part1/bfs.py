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


def bfs(graph, start_vertex):
    visited = set()
    queue = [start_vertex]
    bfs_order = []
    visited.add(start_vertex)
    while queue:
        vertex = queue.pop(0)
        bfs_order.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return bfs_order


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
num_vertices = 20
num_edges = 25

# Generate the graph
G = generate_graph(num_vertices, num_edges)
print("Generated Graph:", G.edges())

# Perform BFS
start_vertex = 1
bfs_order = bfs(G, start_vertex)
print("BFS Order starting from vertex", start_vertex, ":", bfs_order)

# Visualize the graph with BFS path highlighted
visualize_graph(G, bfs_order)
