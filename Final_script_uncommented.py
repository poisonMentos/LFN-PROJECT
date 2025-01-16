import scipy.io
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import sys 
import os 
import pandas as pd
import random 
import time 
def get_graph(path):
    # Replace 'path_to_file.mtx' with the path to your MTX file
    sparse_matrix = scipy.io.mmread(path)

    Graph = nx.from_scipy_sparse_array(sparse_matrix)

    return Graph

def save_graph(Graph, name): 
    adj_matrix = nx.adjacency_matrix(Graph)
    mmwrite(name, adj_matrix)

def visualize_graph(Graph, save_path):
    plt.figure(figsize=(8, 6))  # Set the figure size
    pos = nx.spring_layout(Graph)  # positions for all nodes

    # Get degrees of each node
    degree_dict = dict(Graph.degree())

    # Assign colors based on degree
    node_color = [degree_dict[node] for node in Graph.nodes()]

    # Draw the graph with node labels
    nx.draw(Graph, pos, node_color=node_color, node_size=80, with_labels=True)

    # Set the title of the plot
    plt.title('Graph Visualization')

    # Save the plot to the specified path
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()  # Close the figure to free up memory

def get_degrees(Graph): 
    degrees = dict(Graph.degree())

    return degrees

def get_average_node_degree(Graph): 
    degrees = dict(get_degrees(Graph))
    average_degree = sum(degrees.values()) / len(degrees)

    return average_degree

def get_paradox_vector(Graph):
    degrees = dict(get_degrees(Graph))
    average_degree = sum(degrees.values()) / len(degrees)

    binary_vector = [1 if degrees[node] < average_degree else 0 for node in Graph.nodes()]

    return binary_vector

def calculate_paradox_percentage(Graph): 
    paradox_vector = get_paradox_vector(Graph)
    paradox_percentage = (paradox_vector.count(1) / len(paradox_vector)) * 100

    return paradox_percentage

def average_neighbor_degree(Graph):
    avg_neighbor_degrees = {}
    for node in Graph.nodes():
        neighbors = list(Graph.neighbors(node))
        if neighbors:  # Check if the node has neighbors
            neighbor_degrees = [Graph.degree(neighbor) for neighbor in neighbors]
            avg_neighbor_degrees[node] = sum(neighbor_degrees) / len(neighbor_degrees)
        else:
            avg_neighbor_degrees[node] = 0  # If no neighbors, average degree is 0
    return avg_neighbor_degrees

def get_paradox_vector_neighbor(G): 
    avg_neighbor_degrees = average_neighbor_degree(G)

    # Calculate the binary vector
    binary_vector = []
    for node in G.nodes():
        node_degree = G.degree(node)
        avg_neighbor_degree = avg_neighbor_degrees[node]
        if node_degree < avg_neighbor_degree:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    
    return binary_vector

def calculate_neighbor_paradox_percentage(Graph): 
    neighbor_based_paradox_vector = get_paradox_vector_neighbor(Graph)
    paradox_percentage_neighbor = (neighbor_based_paradox_vector.count(1) / len(neighbor_based_paradox_vector)) * 100

    return paradox_percentage_neighbor

def compute_eigenvector_centrality(Graph): 
    eigenvector_centrality = nx.eigenvector_centrality(Graph)
    
    return eigenvector_centrality

def compute_closeness_centrality(Graph):
    closeness_centrality = nx.closeness_centrality(Graph)

    return closeness_centrality

def approximate_closeness_centrality(G, num_landmarks=500):
    nodes = list(G.nodes())
    if num_landmarks > len(nodes):
        num_landmarks = len(nodes)

    # Randomly select landmarks
    landmarks = random.sample(nodes, num_landmarks)

    # Distance from each landmark to all other nodes
    landmark_distances = {}
    for landmark in landmarks:
        distances = nx.single_source_shortest_path_length(G, landmark)
        for node, distance in distances.items():
            if node in landmark_distances:
                landmark_distances[node].append(distance)
            else:
                landmark_distances[node] = [distance]

    # Initialize centrality dict to handle isolated nodes
    closeness_centrality = {node: 0 for node in G}  # Default to 0 for nodes not reached

    # Calculate approximate closeness centrality
    for node, distances in landmark_distances.items():
        sum_distances = sum(distances)
        reachable_landmarks = len(distances)
        if sum_distances > 0:
            # Normalize by the number of nodes minus one that are reachable
            n = G.number_of_nodes()
            closeness_centrality[node] = reachable_landmarks / sum_distances * (n-1) / (reachable_landmarks-1 if reachable_landmarks > 1 else 1)
        else:
            # Handle isolated nodes by leaving centrality as 0 or setting a specific rule
            closeness_centrality[node] = 0

    return closeness_centrality

def approximate_eigenvector_centrality(G, max_iter=100, tol=1e-6):
    # Initialize the vector with all ones
    n = len(G)
    centrality = np.ones(n)

    # Convert graph to adjacency matrix as a NumPy array
    A = nx.to_numpy_array(G)

    for _ in range(max_iter):
        # Matrix-vector multiplication
        next_centrality = A @ centrality
        
        # Normalizing the vector
        next_centrality = next_centrality / np.linalg.norm(next_centrality)

        # Check for convergence
        if np.linalg.norm(next_centrality - centrality) < tol:
            break
        
        centrality = next_centrality

    return dict(zip(G.nodes, centrality))

def process_data_mode(G, output_folder, filename):
    data_records = []

    # Calculate centrality measures
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)

    # Get average neighbor degree
    avg_neighbor_degrees = average_neighbor_degree(G)

    for node in G.nodes():
        data_records.append({
            'Node': node,
            'Node Degree': G.degree(node),
            'Average Neighbor Node Degree': avg_neighbor_degrees[node],
            'Closeness Centrality': closeness_centrality[node],
            'Eigenvector Centrality': eigenvector_centrality[node]
        })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data_records)
    csv_path = os.path.join(output_folder, filename + '_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data for {filename} saved to {csv_path}")

def approximate_data_mode(G, output_folder, filename):
    data_records = []
    # Time the computation of approximate closeness centrality
    start_time = time.time()
    approximate_closeness_centralities = approximate_closeness_centrality(G)
    closeness_time = time.time() - start_time
    print("Time to compute approximate closeness centrality:", closeness_time, "seconds")

    # Time the computation of approximate eigenvector centrality
    start_time = time.time()
    approximate_eigenvector_centralities = approximate_eigenvector_centrality(G)
    eigenvector_time = time.time() - start_time
    print("Time to compute approximate eigenvector centrality:", eigenvector_time, "seconds")
    # Get average neighbor degree
    avg_neighbor_degrees = average_neighbor_degree(G)

    for node in G.nodes():
        data_records.append({
            'Node': node,
            'Node Degree': G.degree(node),
            'Average Neighbor Node Degree': avg_neighbor_degrees[node],
            'Closeness Centrality': approximate_closeness_centralities[node],
            'Eigenvector Centrality': approximate_eigenvector_centralities[node],
        })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data_records)
    csv_path = os.path.join(output_folder, filename + '_approximate_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Approximate data for {filename} saved to {csv_path}")

    return closeness_time, eigenvector_time

def main():
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <path_to_mtx_folder> <output_folder> [mode]")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "default"

    # Ensure the output folder exists, create if it does not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize DataFrame
    data_records = []
    time_records = []
    # Process each mtx file in the directory
    for file in os.listdir(input_folder):
        if file.endswith(".mtx"):
            full_file_path = os.path.join(input_folder, file)
            G = get_graph(full_file_path)

            filename_without_extension = os.path.splitext(file)[0]
            full_path_to_save = os.path.join(output_folder, filename_without_extension + '.png')

            if mode == "figures":
                visualize_graph(G, full_path_to_save)
                print(f"Processed and saved visualization for {file}")
            elif mode == "stat":
                # Calculate paradox percentages
                paradox_percentage = calculate_paradox_percentage(G)
                paradox_percentage_neighbor = calculate_neighbor_paradox_percentage(G)

                # Append record to the data list
                data_records.append({
                    'Filename': filename_without_extension,
                    'Paradox Percentage': paradox_percentage,
                    'Neighbor Paradox Percentage': paradox_percentage_neighbor
                })
                print("Stat data processed for: ", filename_without_extension)
            elif mode == "data":
                process_data_mode(G, output_folder, filename_without_extension)
            elif mode == "Approx": 
                
                closeness_time, eigenvector_time = approximate_data_mode(G, output_folder, filename_without_extension)
                
                # Store times in data records
                time_records.append({
                    'Filename': file,
                    'Closeness Centrality Time (s)': closeness_time,
                    'Eigenvector Centrality Time (s)': eigenvector_time
                })

                

    # If in stat mode, save data to CSV
    if mode == "stat" and data_records:
        df = pd.DataFrame(data_records)
        csv_path = os.path.join(output_folder, "paradox_percentages.csv")
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

    if mode == "Approx" and time_records:
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(time_records)
        csv_path = os.path.join(output_folder, 'approximate_times.csv')
        df.to_csv(csv_path, index=False)
        print(f"Approximate times for {file} saved to {csv_path}")
    
if __name__ == "__main__":
    main()

