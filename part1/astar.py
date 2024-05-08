import networkx as nx
import matplotlib.pyplot as plt
import random
import heapq

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

def heuristic(u, v):
    return 0

def a_star(graph, start, goal):
    pq = []
    heapq.heappush(pq, (0, start, []))  # (cost, vertex, path)
    visited = set()
    while pq:
        (cost, current_vertex, path) = heapq.heappop(pq)
        if current_vertex in visited:
            continue
        visited.add(current_vertex)
        path = path + [current_vertex]

        if current_vertex == goal:
            return path

        for neighbor, attrs in graph[current_vertex].items():
            if neighbor not in visited:
                weight = attrs['weight']
                heapq.heappush(pq, (cost + weight, neighbor, path))

    return []

def visualize_graph(graph, path):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=1)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    edge_labels = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=2)

    plt.title("Path found by A*")
    plt.axis('off')
    plt.show()

# Graph parameters
num_vertices = 10
num_edges = 15

# Generate the weighted graph
G = generate_weighted_graph(num_vertices, num_edges)
start_vertex = 1
goal_vertex = num_vertices

# Calculate path using A* algorithm
path = a_star(G, start_vertex, goal_vertex)
print("Path found by A* from vertex", start_vertex, "to vertex", goal_vertex, ":", path)

# Visualize the graph with A* path highlighted
visualize_graph(G, path)