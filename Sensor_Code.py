import os
import numpy as np
from scipy.io import loadmat

# Directory where the .mat files are stored
directory = 'C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\HRI_Affectiva_Recordings'

participants = [f"U{str(i).zfill(3)}" for i in range(1, 19)]  # This generates U001 to U018

nodes = [261,53,55,127,22,32,94,59,44,67,132,68,66,91,68,44,62,63]  # Number of nodes for each participant

min_vector_size = float('inf')
#amk = [1,2,3]
data = {}  # must be to store temperature and skin vectors for each participant

for participant, num_nodes in zip(participants, nodes):
    temperature = []
    skin = []

    for filename in os.listdir(directory):
        if filename.endswith(".mat") and participant in filename:
            file_path = os.path.join(directory, filename)
            mat = loadmat(file_path)  # load mat-file

            cel_data = mat['data']['cel'][0,0]  # directly access to 'cel' data
            eda_data = mat['data']['eda'][0,0]  # directly access to 'eda' data

            temperature.extend(cel_data.flatten())  # append temperature data
            skin.extend(eda_data.flatten())  # append skin conductance data

    # Calculate the number of features per node
    features_per_node = len(temperature) // num_nodes

    # Split the 'temperature' and 'skin' lists into vectors of size 'features_per_node'
    temperature_vectors = [temperature[i:i+features_per_node] for i in range(0, len(temperature), features_per_node)]
    skin_vectors = [skin[i:i+features_per_node] for i in range(0, len(skin), features_per_node)]

    # Discard the last vector if it's smaller than 'features_per_node'
    if len(temperature_vectors[-1]) < features_per_node:
        temperature_vectors = temperature_vectors[:-1]
    if len(skin_vectors[-1]) < features_per_node:
        skin_vectors = skin_vectors[:-1]

    # Store temperature and skin vectors for each participant
    data[participant] = {
        'temperature': temperature_vectors,
        'skin': skin_vectors
    }

    # Update the smallest vector size
    min_vector_size = min(min_vector_size, features_per_node)

    print(f"For participant {participant}:")
    print(f"Primary temperature vector size: {len(temperature)}")
    print(f"Primary skin vector size: {len(skin)}")
    print(f"Number of nodes = {num_nodes}, the number of features in each node = {features_per_node}.\n")


print(f"The smallest size of the created vectors is {min_vector_size}.\n")


# Create a new dictionary to store the downsampled data
data_downsampled_temp = {}
data_downsampled_eda = {}

for participant, num_nodes in zip(participants, nodes):
    # Retrieve the original temperature and skin vectors for the current participant
    original_temperature_vectors = data[participant]['temperature']
    original_skin_vectors = data[participant]['skin']

    # Initialize the downsampled temperature and skin vectors
    downsampled_temperature = []
    downsampled_skin = []

    for t_vec, s_vec in zip(original_temperature_vectors, original_skin_vectors):
        # Calculate the downsampling factor
        downsample_factor = len(t_vec) / min_vector_size

        # Create downsampled vectors
        downsampled_t_vec = [np.mean(t_vec[int(i*downsample_factor):int((i+1)*downsample_factor)]) for i in range(min_vector_size)]
        downsampled_s_vec = [np.mean(s_vec[int(i*downsample_factor):int((i+1)*downsample_factor)]) for i in range(min_vector_size)]
        
        # Append the downsampled vectors to the respective lists
        downsampled_temperature.append(downsampled_t_vec)
        downsampled_skin.append(downsampled_s_vec)

    # Store the downsampled temperature and skin vectors for the current participant
    data_downsampled_temp[participant] = [downsampled_temperature]
    data_downsampled_eda[participant] = [downsampled_skin]

    print(f"For participant {participant}, the number of nodes = {num_nodes}.")
    print(f"Downsampling factor = {downsample_factor}.")
    print(f"Number of features in each node after downsampling = {len(downsampled_temperature[0])}.\n")

# Initialize lists to store the data
temperature_data = []
skin_data = []

# For each participant converting the lists into numpy arrays and append them directly to the data lists
for participant in participants:
    temperature_data.append(np.array(data_downsampled_temp[participant]).squeeze())
    skin_data.append(np.array(data_downsampled_eda[participant]).squeeze())
