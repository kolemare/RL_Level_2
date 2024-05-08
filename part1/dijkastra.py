import networkx as nx
import matplotlib.pyplot as plt
import random


def generate_weighted_graph(num_vertices, num_edges):
    G = nx.Graph()
    G.add_nodes_from(range(1, num_vertices + 1))
    while len(G.edges) < num_edges:
        v1 = random.randint(1, num_vertices)
        v2 = random.randint(1, num_vertices)
        weight = random.randint(1, 10)
        if v1 != v2:
            G.add_edge(v1, v2, weight=weight)
    return G


def dijkstra(graph, start_vertex):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start_vertex] = 0
    pq = [(0, start_vertex)]

    while pq:
        current_distance, current_vertex = min(pq, key=lambda x: x[0])
        pq.remove((current_distance, current_vertex))

        for neighbor, attrs in graph[current_vertex].items():
            weight = attrs['weight']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                pq.append((distance, neighbor))

    return distances


def visualize_graph(graph, distances, start_vertex):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=1)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    edge_labels = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Highlight the shortest path tree from start_vertex
    for vertex in graph:
        path = nx.shortest_path(graph, source=start_vertex, target=vertex, weight='weight')
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=2)

    plt.title(f"Shortest Paths from Vertex {start_vertex}")
    plt.axis('off')
    plt.show()


# Graph parameters
num_vertices = 10
num_edges = 15

# Generate the weighted graph
G = generate_weighted_graph(num_vertices, num_edges)
print("Generated Graph:", G.edges(data=True))

# Calculate shortest paths using Dijkstra's algorithm
start_vertex = 1
distances = dijkstra(G, start_vertex)
print("Shortest Distances from vertex", start_vertex, ":", distances)

# Visualize the graph with shortest paths highlighted
visualize_graph(G, distances, start_vertex)