'''
Networks architectures
'''
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN_LSTM(nn.Module):
    '''Encoder
    Embedding -> Convolutional layer -> One-layer LSTM
    '''
    def __init__(self, input_dim, embedding_dim, ker_size, n_filters, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(ker_size, embedding_dim))
        self.lstm = nn.LSTM(input_size=n_filters, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
   
    def forward(self, text, h_0):
        # CNN and LSTM network
        '''
        At every time step, the model reads one chunk which has a size of 20 words.

        --- input & output dimension ---
        
        Input text: 1 * 20
        
        **Embedding**
        1.Input: 1 * 20
        2.Output: 1 * 20 * 100
        
        **CNN**
        1. Input(minibatch×in_channels×iH×iW): 1 * 1 * 20 * 100
        2. Output(minibatch×out_channels×oH×oW): 1 * 128 * 16 * 1
        
        **LSTM**
        1. Inputs: input, (h_0, c_0)
        input(seq_len, batch, input_size): (16, 1 , 128)
        c_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        h_0(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        2. Outputs: output, (h_n, c_n)
        output:
        h_n(num_layers * num_directions, batch, hidden_size): (1 * 1, 1, 128)
        

        '''
        embedded = self.embedding(text)
        #print(embeded.size())
        conved = self.relu(self.conv(embedded.unsqueeze(1)))  # 1 * 128 * 16 * 1
        #print(conved.size())
        # conv -> relu -> dropout
        batch = conved.size()[0]
        conved = self.dropout(conved)
        conved = conved.squeeze(3)  # 1 * 128 * 16
        conved = torch.transpose(conved, 1, 2)  # 1 * 16 * 128
        conved = torch.transpose(conved, 1, 0)  # 16 * 1 * 128
        c_0 = torch.zeros([1, batch, 128]).to(device)
        output, (hidden, cell) = self.lstm(conved, (h_0, c_0))
        ht = hidden.squeeze(0)  # 1 * 128
        return ht


class Policy_S(nn.Module):
    '''Stopping module

    Three hidden-layer MLP with 128 hidden units per layer.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()    
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
          
    def forward(self, ht):
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)
        return out
    

class Policy_N(nn.Module):
    '''Re-reading and skipping module
    
    Three hidden-layer MLP with 128 hidden units per layer.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()    
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
         
    def forward(self, ht):
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out
    

class Policy_C(nn.Module):
    '''Classifier
    
    Single-layer MLP with 128 hidden units.
    '''

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        #self.fc = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
         
    def forward(self, ht):
        #return self.fc(ht)
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ValueNetwork(nn.Module):
    '''Baseline
    Reduce the variance.

    Single-layer MLP with 128 hidden units.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        #self.fc = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, ht):
        out = self.fc1(ht)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out