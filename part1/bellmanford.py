import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_weighted_graph(num_vertices, num_edges):
    G = nx.DiGraph()
    G.add_nodes_from(range(1, num_vertices + 1))
    while len(G.edges) < num_edges:
        v1 = random.randint(1, num_vertices)
        v2 = random.randint(1, num_vertices)
        weight = random.randint(-10, 10)
        if v1 != v2:
            G.add_edge(v1, v2, weight=weight)
    return G

def bellman_ford(graph, start_vertex):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start_vertex] = 0

    for _ in range(len(graph) - 1):
        for vertex in graph:
            for neighbor, attrs in graph[vertex].items():
                weight = attrs['weight']
                if distances[vertex] + weight < distances[neighbor]:
                    distances[neighbor] = distances[vertex] + weight

    # Check for negative weight cycles
    for vertex in graph:
        for neighbor, attrs in graph[vertex].items():
            weight = attrs['weight']
            if distances[vertex] + weight < distances[neighbor]:
                print("Graph contains negative weight cycle")
                return None  # Return None to indicate the presence of a negative cycle

    return distances

def visualize_graph(graph, distances, start_vertex):
    pos = nx.spring_layout(graph)  # positions for all nodes
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=1)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    edge_labels = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Highlight the shortest path tree from start_vertex
    if distances is not None:
        for vertex in distances:
            if distances[vertex] < float('infinity'):
                try:
                    path = nx.shortest_path(graph, source=start_vertex, target=vertex, weight='weight')
                    path_edges = list(zip(path[:-1], path[1:]))
                    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=2)
                except nx.NetworkXNoPath:
                    continue

    plt.title(f"Shortest Paths from Vertex {start_vertex}")
    plt.axis('off')
    plt.show()

# Graph parameters
num_vertices = 10
num_edges = 15

# Generate the weighted graph
G = generate_weighted_graph(num_vertices, num_edges)
print("Generated Graph:", G.edges(data=True))

# Calculate shortest paths using Bellman-Ford algorithm
start_vertex = 1
distances = bellman_ford(G, start_vertex)
if distances:
    print("Shortest Distances from vertex", start_vertex, ":", distances)

    # Visualize the graph with shortest paths highlighted
    visualize_graph(G, distances, start_vertex)
else:
    print("No visualization due to negative weight cycle.")