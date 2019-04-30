'''
Compute FLOPs of models.

Approximation.
'''
from utils import print_model_parm_flops
from networks import CNN_LSTM, Policy_C, Policy_N, Policy_S, ValueNetwork
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
      
    def __init__(self,input_dim, embedding_dim, ker_size, n_filters, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embedding.weight.requires_grad = False
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(ker_size, embedding_dim))
      
    def forward(self, text): 
        '''
            input 1 by 20
        '''
        embedded = self.embedding(text)
        conved = F.relu(self.conv(embedded.unsqueeze(1))) # input:[minibatch×in_channels×iH×iW] 1*1*20*100
        return conved

class Policy_S(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()    
        self.fc_s_hidden0 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_s_hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_s_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_s_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, ht):
        out = self.relu(self.fc_s_hidden0(ht))
        out = self.relu(self.fc_s_hidden1(out))
        out = self.relu(self.fc_s_hidden2(out))
        out = self.fc_s_output(out)
        out = self.sigmoid(out)
        return out

class Policy_C(nn.Module):
  
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc_c_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_c_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, ht):
        out = self.relu(self.fc_c_hidden(ht))
        out = self.fc_c_output(out)
        return out
  
  
class Policy_N(nn.Module):
  
    def __init__(self, hidden_dim, max_k):
        super().__init__()
        self.fc_n_hidden0 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_n_hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_n_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_n_output = nn.Linear(hidden_dim, max_k)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, ht):
        out = self.relu(self.fc_n_hidden0(ht))
        out = self.relu(self.fc_n_hidden1(out))
        out = self.relu(self.fc_n_hidden2(out))
        out = self.fc_n_output(out)
        return self.softmax(out)


INPUT_DIM = 25002
EMBEDDING_DIM = 100
KER_SIZE = 5
HIDDEN_DIM = 128
OUTPUT_DIM = 1
CHUNCK_SIZE = 20
TEXT_LEN = 400
MAX_K = 3
N_FILTERS = 128
BATCH_SIZE = 1


cnn_model = CNN(INPUT_DIM, EMBEDDING_DIM, KER_SIZE, N_FILTERS, HIDDEN_DIM)
test_policy_s = Policy_S(HIDDEN_DIM, OUTPUT_DIM).train()
test_policy_n = Policy_N(HIDDEN_DIM, MAX_K).train()
test_policy_c = Policy_C(HIDDEN_DIM, OUTPUT_DIM).train()

input_size = torch.randint(1,2, (1, 20))
cnn_cost = print_model_parm_flops(cnn_model, input_size)
p = torch.rand(1,128)
s_cost = print_model_parm_flops(test_policy_s, p)
c_cost = print_model_parm_flops(test_policy_c, p)
n_cost = print_model_parm_flops(test_policy_n, p)
print('cnn_cost', cnn_cost)
print('s_cost', s_cost)
print('c_cost', c_cost)
print('n_cost', n_cost)
