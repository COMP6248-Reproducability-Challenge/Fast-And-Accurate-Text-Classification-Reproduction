'''
The whole reading model.

CNN + LSTM + A classifier with multinomial distribution.
'''

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from torchtext import datasets
from torchtext import data
import os
import time
import numpy as np 
import random
import argparse
from sklearn.metrics import accuracy_score

from networks import CNN_LSTM, Policy_C, Policy_N, Policy_S, ValueNetwork
from utils.utils import sample_policy_c, sample_policy_n, sample_policy_s, evaluate_earlystop, compute_policy_value_losses
from utils.utils import cnn_cost, lstm_cost, c_cost, n_cost, s_cost, cnn_whole

desc = '''
The whole reading model.

CNN + LSTM + A classifier with multinomial distribution.
'''
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 2019)')
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(args.seed)

TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, fix_length=400) # 
LABEL = data.LabelField(dtype=torch.float)

print('Splitting data...')
# download the IMDB dataset
train, test_data = datasets.IMDB.splits(TEXT, LABEL) # 25,000 training and 25,000 testing data
train_data, valid_data = train.split(split_ratio=0.8) # split training data into 20,000 training and 5,000 vlidation sample

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

MAX_VOCAB_SIZE = 25000

# use pretrained embedding of glove
print('Building vocabulary...')
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# split the datasets into batches
BATCH_SIZE = 64  # the batch size for a dataset iterator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
print('Building iterators...')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    device=device)

# set up parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
KER_SIZE = 5
HIDDEN_DIM = 128
LABEL_DIM = 2
N_FILTERS = 128
learning_rate = 0.001

# the number of training epoches
num_of_epoch = 10

# set up the criterion
criterion = nn.CrossEntropyLoss().to(device)
# set up models
clstm = CNN_LSTM(INPUT_DIM, EMBEDDING_DIM, KER_SIZE, N_FILTERS, HIDDEN_DIM).to(device)
policy_c = Policy_C(HIDDEN_DIM, HIDDEN_DIM, LABEL_DIM).to(device)

# set up optimiser
params = list(clstm.parameters()) + list(policy_c.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

# add pretrained embeddings
pretrained_embeddings = TEXT.vocab.vectors
clstm.embedding.weight.data.copy_(pretrained_embeddings)
clstm.embedding.weight.requires_grad = True  # update the initial weights


def evaluate(iterator):
    clstm.eval()
    policy_c.eval()
    true_labels = []
    pred_labels = []
    eval_loss = 0
    for index, valid in enumerate(iterator):
        label = valid.label
        text = valid.text.transpose(0,1)
        ht = clstm(text)
        label_raws = policy_c(ht)
        label_probs = F.softmax(label_raws, dim=1)
        m = Categorical(label_probs)
        pred_label = m.sample()
        true_labels.extend(label.cpu().numpy())
        pred_labels.extend(pred_label.cpu().squeeze().numpy())
        loss = criterion(label_raws.squeeze(), label.to(torch.long))
        eval_loss += loss/len(iterator)
    eval_accuracy = accuracy_score(true_labels, pred_labels)
    return eval_loss, eval_accuracy

def main():
    '''
    Training and evaluation of the model.
    '''
    print('training starts...')
    for epoch in range(num_of_epoch):
        clstm.train()
        policy_c.train()
        true_labels = []
        pred_labels = []
        train_loss = 0
        for index, train in enumerate(train_iterator):
            label = train.label              # output_dim:64
            text = train.text.transpose(0,1) #: 64, 400
            ht = clstm(text)                 #: 64, 128
            label_raws = policy_c(ht)
            optimizer.zero_grad()
            loss = criterion(label_raws.squeeze(), label.to(torch.long))
            loss.backward()
            optimizer.step()
            # draw a prediction label
            label_probs = F.softmax(label_raws.detach(), dim=1)
            m = Categorical(label_probs)
            pred_label = m.sample()
            true_labels.extend(label.cpu().numpy())
            pred_labels.extend(pred_label.cpu().squeeze().numpy())
            train_loss += loss/len(train_iterator)
        train_accuracy = accuracy_score(true_labels, pred_labels)
        print('epoch:{0}, train accuracy:{1}, train_loss:{2}'.format(epoch, train_accuracy, train_loss))
        eval_loss, eval_accuracy = evaluate(valid_iterator)
        print('epoch:{0}, eval accuracy:{1}, eval_loss:{2}'.format(epoch, eval_accuracy, eval_loss))
    # testing
    test_loss, test_accuracy = evaluate(test_iterator)
    print('\n Test accuracy:{1}, test loss:{2}'.format(epoch, test_accuracy, test_loss))



if __name__ == '__main__':
    main()
    cost = cnn_whole + c_cost + lstm_cost * 24
    print('whole reading FLOPs per data: ', cost)