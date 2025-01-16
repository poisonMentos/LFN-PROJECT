import networkx as nx
import numpy as np
import os
import sys 
import math
from scipy.io import mmread  # To read mtx files
from scipy.io import mmwrite

# Function to generate and save random graphs as .mtx files
def save_random_graphs_as_mtx(node_range, edge_probabilities, num_graphs=8, directory='graph_files'):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Generate node counts from a normal distribution
    mean, std = np.mean(node_range)/2, np.std(node_range)/2
    node_counts = np.random.normal(mean, std, num_graphs)
    node_counts = np.clip(node_counts, node_range[0]/2, node_range[1]/2).astype(int)

    print(node_counts)

    for idx, nodes in enumerate(node_counts):
        if idx < len(edge_probabilities):
            # Use the computed probability for the Erdős-Rényi graph from the analysis
            p = edge_probabilities[idx]
        else:
            # Default to a reasonable probability if we run out of pre-computed probabilities
            p = 0.5

        # Generate an Erdős-Rényi graph
        print("Generating Erdős-Rényi graph: ", f'erdos_renyi_{idx+1}_n{nodes}_p{p:.2f}.mtx', " with ", nodes, " nodes")
        G_er = nx.erdos_renyi_graph(nodes, p)
        mmwrite(os.path.join(directory, f'erdos_renyi_{idx+1}_n{nodes}_p{p:.2f}.mtx'), nx.adjacency_matrix(G_er))
        print("Saved Erdős-Rényi graph: ", f'erdos_renyi_{idx+1}_n{nodes}_p{p:.2f}.mtx')

        # Generate a Watts-Strogatz graph
        k = max(2, nodes // 4)  # Degree of nodes
        rewiring_prob = p  # Using the same probability for simplicity
        print("Generating Watts-Strogatz graph: ", f'watts_strogatz_{idx+1}_n{nodes}_k{k}_p{rewiring_prob:.2f}.mtx', " with ", nodes, " nodes")
        G_ws = nx.watts_strogatz_graph(nodes, k, rewiring_prob)
        mmwrite(os.path.join(directory, f'watts_strogatz_{idx+1}_n{nodes}_k{k}_p{rewiring_prob:.2f}.mtx'), nx.adjacency_matrix(G_ws))
        print("Saved Watts-Strogatz graph: ", f'watts_strogatz_{idx+1}_n{nodes}_k{k}_p{rewiring_prob:.2f}.mtx')

        # Generate a Barabási-Albert graph
        m = max(1, int(p * nodes))  # Number of edges to attach from a new node to existing nodes
        print("Generating Barabási-Albert graph: ", f'barabasi_albert_{idx+1}_n{nodes}_m{m}.mtx', " with ", nodes, " nodes")
        G_ba = nx.barabasi_albert_graph(nodes, m)
        mmwrite(os.path.join(directory, f'barabasi_albert_{idx+1}_n{nodes}_m{m}.mtx'), nx.adjacency_matrix(G_ba))
        print("Saved Barabási-Albert graph: ", f'barabasi_albert_{idx+1}_n{nodes}_m{m}.mtx')

def analyze_node_distribution(directory):
    node_counts = []
    edge_counts = []
    edge_probabilities = []

    # Loop through all the .mtx files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.mtx'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Read the matrix file
            graph_matrix = mmread(file_path)
            
            # Convert the matrix to a graph
            G = nx.from_scipy_sparse_array(graph_matrix)
            
            # Get the number of nodes and edges
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            node_counts.append(num_nodes)
            edge_counts.append(num_edges)

            # Calculate the probability of an edge existing
            total_possible_edges = num_nodes * (num_nodes - 1) / 2
            edge_probability = num_edges / total_possible_edges if total_possible_edges > 0 else 0
            edge_probabilities.append(edge_probability)

            print("Loaded file: ", filename, "with", num_nodes, "nodes and", num_edges, "edges.")

    return node_counts, edge_counts, edge_probabilities


def main():
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <input_folder< <output_folder> <number_of_graphs>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    number_of_graphs = int(sys.argv[3])

    node_counts, edge_counts, edge_probabilities = analyze_node_distribution(input_folder)
    print("Finished analysis of graphs")
    iteration_counter = math.floor(number_of_graphs/3)
    # Define the range of number of nodes
    node_range = (min(node_counts), max(node_counts))
    # Save 8 graphs for each method in the specified directory
    save_random_graphs_as_mtx(node_range, edge_probabilities, iteration_counter, output_folder)
    print("All command finished succesfully")

if __name__ == "__main__":
    main()