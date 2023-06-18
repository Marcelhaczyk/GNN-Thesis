import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize #sent_tokenize --> tokenizacja na zdania, word_tokeniza --> na słowa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import random
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
from torch_geometric.nn import global_mean_pool
from Sensor_Code import temperature_data, skin_data
from torch.nn import L1Loss


# Download necessary NLTK data
nltk.download('punkt') # We download a model that contains a set of rules and patterns that the Punkt tokenizer uses to split text into sentences. For example, it is responsible to proccess a text such as in 'Mr.Haczyk' dot is not the meaning of the end of the sentence
nltk.download('stopwords') #This line downloads a list of common stopwords in English, such as "the," "a," "and," and so on.
nltk.download('wordnet') #This line downloads the WordNet lexical database, which is simply a large, structured collection of words and their meanings. 

# Function which returns a list with lemmatized words without stop words.
def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer() #lemmatization
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence) #Tokenizacja słów 
        filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word.isalnum()] #Assigns the resulting list of filtered and lemmatized words to a new variable called filtered_words. Steps: Lemmatization, change to lower letter, check if this is not a stop word and if word alphanumeric
        preprocessed_sentences.append(" ".join(filtered_words))
    return preprocessed_sentences 
# preprocessed_sentences = ['hi nao', 'left perhaps', 'could please stand'] ['said left', ''],['well thank', 'nao', 'name'],['name sally'] tekst wszystkich 18-participants (parts of conversation)


def preprocess_dialogue(text):
    turns = text.split('\n')
    preprocessed_turns = []
    for turn in turns:
        preprocessed_turns.extend(preprocess_text(turn))
    return preprocessed_turns
# prepocessed_turns = ['left', 'perhaps', 'could please stand', 'well thank', 'nao', 'name', 'name hawier', 'well good bring face closer'] zmienna prepocessed_turns jest zawsze z jednego participanta (whole conversation)

file_paths = [ 'C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU001.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU002.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU003.txt',
              'C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU004.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU005.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU006.txt',
              'C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU007.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU008.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU009.txt',
              'C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU010.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU011.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU012.txt',
              'C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU013.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU014.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU015.txt',
              'C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU016.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU017.txt','C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\Transcripts\TranscriptU018.txt']


# Preprocessing turns
all_preprocessed_turns = []
for file_path in file_paths:
    transcript = open(file_path, 'r').read()
    all_preprocessed_turns.extend(preprocess_dialogue(transcript))
# all_preprocessed_turns = ['hi everyone nao', 'hello hello', 'right could please stand', 'thank', 'okay thank', ...] Whole 18 particiapnts turns (size-1408)

from PersonalityLabelsCode import participants_labels
print(participants_labels)


#Creating the adjacency matrix
def adjacency_matrix(similarity_matrix, threshold):
    adj_matrix = np.copy(similarity_matrix)
    adj_matrix[adj_matrix < threshold] = 0 # Sets all values in the adj.matrix that are less than threshold to 0
    np.fill_diagonal(adj_matrix, 0) #cosine similarity between a turn and itself is always 0
    return adj_matrix

threshold = 0.3

# Create a TF-IDF vectorizer and fit it to the preprocessed turns for all transcripts
vectorizer = TfidfVectorizer()
vectorizer.fit(all_preprocessed_turns)
#print(vectorizer.vocabulary_)  # prints the learned vocabulary
#print(len(vectorizer.vocabulary_))  # prints the size of the vocabulary
# Store the transformed TF-IDF matrices for each dialogue. (1 array króry przechowuje 18 arrays i w kazdym z nich liczby ktore są wagami slów które są używane) 
# Np w pierwszym grafie, rozmiar to 261x543 mimo ze jak rozwiniemy to zawiera on 758 liczb, ale w tym 758 są duplikaty, a 256 to bez duplikatów. Używamy 543 dlatego że to liczba wszystkich unique words w calym datasecie.
tfidf_matrices = []
for file_path in file_paths:
    transcript = open(file_path, 'r').read()
    preprocessed_turns = preprocess_dialogue(transcript)
    tfidf_matrix = vectorizer.transform(preprocessed_turns)
    tfidf_matrices.append(tfidf_matrix)
#print(tfidf_matrices)

mae_criterion = L1Loss()

num_of_participants = 18
pyg_graphs = []
for i in range(0,num_of_participants):
    tfidf_matrix = tfidf_matrices[i]

    # Create the similarity matrix, we calculate how similar each pair of turns in the dialogue is - e.g. turn 1 with turn 45. (computes the cosine similarity between each pair of turns in the dialogue)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Create the adjacency matrix (same as similarity matrix but with treshold applied)
    adj_matrix = adjacency_matrix(similarity_matrix, threshold)

    # It converts the adjacency matrix into an edge_index tensor, a PyTorch Geometric representation of edges in a graph.
    edge_index = torch.tensor(np.vstack(np.nonzero(adj_matrix)), dtype=torch.long) #It contains info about egdes, i mean which node is connected to which node.

    #Creating the labels (for 1 graph)
    my_labels = participants_labels[i]

    #Add temperature features
    temp_matrix = temperature_data[i]
    eda_matrix = skin_data[i]
    #temp_matrix = np.reshape(temp_matrix, (-1, 1))
    feature_matrix = np.concatenate((tfidf_matrix.toarray(), temp_matrix, eda_matrix), axis=1)

    #Creating a graph
    data = Data(x=torch.tensor(feature_matrix, dtype=torch.float), edge_index=edge_index, y=torch.tensor(my_labels, dtype=torch.float).unsqueeze(0))
    # Add the Data object to the list of graphs
    pyg_graphs.append(data)

print(pyg_graphs)



# Define the GNN model
class SimpleGNN(nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(1081, 32)
        self.conv2 = GCNConv(32, 8)
        #self.conv3 = GCNConv(16, 8)
        self.fc = nn.Linear(8, 5)
    
    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.15, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.15, training=self.training)

        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, p=0.15, training=self.training)

        batch = torch.zeros(x.size(0), dtype=torch.long)  
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)

        return x


### Print informations about all graphs ######
def print_graph_info(data):
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of features per node:", data.num_features)
    print("Node Feature Vector:")

    node_id = 0
    node_features = data.x[node_id]
    print('Node 21 Feature Vector',node_features)
    print(len(node_features))


##### TRAINING AND EVALUATION #####
def train(model, data_loader, optimizer, criterion, mae_criterion):
    model.train()
    total_loss = 0.0
    total_mae = 0.0

    for data in data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        mae = mae_criterion(out, data.y)
        total_mae += mae.item()
    return total_loss / len(data_loader), total_mae / len(data_loader)

def evaluate(model, data_loader, criterion, mae_criterion):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            #data = data.to(device)
            out = model(data.x, data.edge_index)          
            loss = criterion(out, data.y)
            total_loss += loss.item()
            mae = mae_criterion(out, data.y)
            total_mae += mae.item()
            all_outputs.append(out)
            all_labels.append(data.y[0])     
    return total_loss / len(data_loader), total_mae / len(data_loader),all_outputs, all_labels


import graphviz
from torchview import draw_graph
graphviz.set_jupyter_format('png')

def main():
    #pyg_graphs = convert_spektral_to_pyg(pyg_graphs)
    for i, data in enumerate(pyg_graphs):
        print(f"Graph {i + 1} Information:")
        print_graph_info(data)
        print()
    model = SimpleGNN()
    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Calculate the lengths of splits
    total = len(pyg_graphs)
    train_len = int(train_ratio * total)
    val_len = int(val_ratio * total)

    # Split the dataset
    train_graphs = pyg_graphs[:train_len]
    val_graphs = pyg_graphs[train_len:train_len + val_len]
    test_graphs = pyg_graphs[train_len + val_len:]

    # Create DataLoaders for training, validation, and test sets
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=True)

    # Set up the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = MSELoss()

# Train the model
    num_epochs = 500
    train_losses = []  #  to store training losses
    test_losses = []  #  to store validation losses

    train_maes = []  # to store training MAEs
    test_maes = []  # to store test MAEs
    for epoch in range(num_epochs):

        train_loss, train_mae = train(model, train_loader, optimizer, criterion, mae_criterion)
        train_losses.append(train_loss)  # to store the current training loss       
        train_maes.append(train_mae)
        test_loss,test_mae, all_outputs, all_labels = evaluate(model, test_loader, criterion, mae_criterion)
        test_losses.append(test_loss)  # to store the current validation loss
        test_maes.append(test_mae)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

    #Print train and validation loss
    for i, (output, labels) in enumerate(zip(all_outputs, all_labels)):
        print(f"Graph {i + 1} Predictions:")
        print(output)
        print(f"Graph {i + 1} Ground truth:")
        print(labels)
        print()
    

    # Plotting training and validation MSEs
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(np.arange(0, num_epochs), test_losses, label="Test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.ylim()  # Change the values to your desired range
    plt.title("Train and Test MSEs")
    plt.show()

    # Plotting Training and Validation MAEs
    plt.figure(figsize=(12, 6))
    plt.plot(train_maes, label="Train MAE")
    plt.plot(test_maes, label="Test MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.ylim()
    plt.title("Train and Test MAEs")
    plt.show()

    # Plot predictions and ground truth
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(output[0])), output[0], label="Predictions")
    plt.scatter(range(len(labels)), labels, label="Ground Truth")
    plt.xlabel("Node Index")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"Graph {i + 1}: Predictions vs Ground Truth")
    plt.ylim(0, 20) 
    plt.show()
    


if __name__ == "__main__":
    main()



def save_model_to_file(model, type_of_model):
    model_file = 'model_' + type_of_model + '.pth'

    torch.save(model.state_dict(), model_file)

    print('Saved neural network model to file {}'.format(model_file))


def load_model_from_files(model_file):
    loaded_model = SimpleGNN() 
    loaded_model.load_state_dict(torch.load(model_file))

    return loaded_model

# Saving the model
save_model_to_file(SimpleGNN(), 'GNN')

# Loading the model
loaded_model = load_model_from_files('model_GNN.pth')


#model_graph_1 = draw_graph(SimpleGNN(), input_size=(261, 1081))
#model_graph_1.visual_graph


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/gnn_model')
data = data.to('cpu')  
model = SimpleGNN().to('cpu')  

writer.add_graph(SimpleGNN(), [data.x, data.edge_index])
writer.close()

