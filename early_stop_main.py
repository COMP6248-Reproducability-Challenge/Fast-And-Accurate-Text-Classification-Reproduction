'''
The early stopping model with only a stopping module.

Use REINFORCE with baseline.
Use discounted rewards for an episode.
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

from networks import CNN_LSTM, Policy_C, Policy_N, Policy_S, ValueNetwork
from utils.utils import sample_policy_c, sample_policy_n, sample_policy_s, evaluate_earlystop, compute_policy_value_losses
from utils.utils import cnn_cost, clstm_cost, c_cost, n_cost, s_cost

desc = '''
The early stopping model with only a stopping module.

Use REINFORCE with baseline.
Use discounted rewards for an episode.
'''
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--alpha', type=float, default=0.2, metavar='A',
                    help='a trade-off parameter between accuracy and efficiency (default: 0.2)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
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
BATCH_SIZE = 1  # the batch size for a dataset iterator
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
OUTPUT_DIM = 1
CHUNCK_SIZE = 20
MAX_K = 4  # the output dimension for step size 0, 1, 2, 3
LABEL_DIM = 2
N_FILTERS = 128
BATCH_SIZE = 1
gamma = args.gamma
alpha = args.alpha
learning_rate = 0.001

# the number of training epoches
num_of_epoch = 10
# the number of batch size for gradient descent when training
batch_sz = 64

# set up the criterion
criterion = nn.CrossEntropyLoss().to(device)
# set up models
clstm = CNN_LSTM(INPUT_DIM, EMBEDDING_DIM, KER_SIZE, N_FILTERS, HIDDEN_DIM).to(device)
policy_s = Policy_S(HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
policy_c = Policy_C(HIDDEN_DIM, HIDDEN_DIM, LABEL_DIM).to(device)
value_net = ValueNetwork(HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)


# set up optimiser
params_pg = list(policy_s.parameters()) + list(policy_c.parameters())
optim_loss = optim.Adam(clstm.parameters(), lr=learning_rate)
optim_policy = optim.Adam(params_pg, lr=learning_rate)
optim_value = optim.Adam(value_net.parameters(), lr=learning_rate)

# add pretrained embeddings
pretrained_embeddings = TEXT.vocab.vectors
clstm.embedding.weight.data.copy_(pretrained_embeddings)
clstm.embedding.weight.requires_grad = True  # update the initial weights

def finish_episode(policy_loss_sum, encoder_loss_sum, baseline_value_batch):
    '''
    Called when a data sample has been processed.
    '''
    baseline_value_sum = torch.stack(baseline_value_batch).sum()
    policy_loss = torch.stack(policy_loss_sum).mean()
    encoder_loss = torch.stack(encoder_loss_sum).mean()
    objective_loss = encoder_loss - policy_loss + baseline_value_sum
    # set gradient to zero
    optim_loss.zero_grad()
    optim_policy.zero_grad()
    optim_value.zero_grad()
    # back propagation
    objective_loss.backward()
    # gradient update
    optim_loss.step()
    optim_policy.step()
    optim_value.step()


def main():
    '''
    Training and evaluation of the model.
    '''
    print('Training starts...')
    for epoch in range(num_of_epoch):
        print('\nEpoch', epoch+1)
        # log the start time of the epoch
        start = time.time()
        clstm.train()
        policy_c.train()
        policy_s.train()
        policy_loss_sum = []
        encoder_loss_sum = []
        baseline_value_batch = []
        for index, train in enumerate(train_iterator):
            label = train.label.to(torch.long) # 64
            text = train.text.view(CHUNCK_SIZE, BATCH_SIZE, CHUNCK_SIZE) # transform 1*400 to 20*1*20
            curr_step = 0
            # set up the initial input for lstm
            h_0 = torch.zeros([1,1,128]).to(device) 
            c_0 = torch.zeros([1,1,128]).to(device)
            saved_log_probs = []
            baseline_value_ep = []
            cost_ep = []   # collect the computational costs for every time step
            while (curr_step < 20):
                '''
                loop until stop decision equals 1 
                or the whole text has been read
                '''
                # read a chunk
                text_input = text[curr_step]
                # hidden state
                ht, ct = clstm(text_input, h_0, c_0)  # 1 * 128
                h_0 = ht.unsqueeze(0).to(device)  # 1 * 1 * 128, next input of lstm
                c_0 = ct
                # compute a baseline value for the value network
                ht_ = ht.clone().detach().requires_grad_(True).to(device)
                bi = value_net(ht_)
                # draw a stop decision
                stop_decision, log_prob_s = sample_policy_s(ht, policy_s)
                stop_decision = stop_decision.item()
                if stop_decision == 1:
                    break
                else:
                    curr_step += 1
                    if curr_step < 20:
                        # If the code can still execute the next loop, it is not the last time step.
                        cost_ep.append(clstm_cost + s_cost)
                        # add the baseline value
                        saved_log_probs.append(log_prob_s)
                        baseline_value_ep.append(bi)
                    
            # add the baseline value at the last step
            baseline_value_ep.append(bi)
            cost_ep.append(clstm_cost + s_cost + c_cost)
            # output of classifier       
            output_c = policy_c(ht)  # classifier
            # compute cross entropy loss
            loss = criterion(output_c, label)
            encoder_loss_sum.append(loss)
            # draw a predicted label 
            pred_label, log_prob_c = sample_policy_c(output_c)
            saved_log_probs.append(log_prob_c.unsqueeze(0) + log_prob_s)
            # compute the policy losses and value losses for the current episode
            policy_loss_ep, value_losses = compute_policy_value_losses(cost_ep, loss, saved_log_probs, baseline_value_ep, alpha, gamma)
            policy_loss_sum.append(torch.cat(policy_loss_ep).sum())
            baseline_value_batch.append(torch.cat(value_losses).sum())
            # Backward and optimize
            if (index + 1) % batch_sz == 0:
                finish_episode(policy_loss_sum, encoder_loss_sum, baseline_value_batch)
                del policy_loss_sum[:], encoder_loss_sum[:], baseline_value_batch[:]  
            
            # print log
            if (index + 1) % 2000 == 0:
                print(f'\n current episode: {index + 1}')
                # log the current position of the text which the agent has gone through
                print('curr_step: ', curr_step)
        print('Epoch time elapsed: %.2f s' % (time.time() - start))
        count_all, count_correct = evaluate_earlystop(clstm, policy_s, policy_c, valid_iterator)
        print('Epoch: %s, Accuracy on the validation set: %.2f' % (epoch + 1, count_correct / count_all))
        count_all, count_correct = evaluate_earlystop(clstm, policy_s, policy_c, train_iterator)
        print('Epoch: %s, Accuracy on the training set: %.2f' % (epoch + 1, count_correct / count_all))
    
    print('Compute the accuracy on the testing set...')
    count_all, count_correct = evaluate_earlystop(clstm, policy_s, policy_c, test_iterator)
    print('Accuracy on the testing set: %.2f' % (count_correct / count_all))


if __name__ == '__main__':
    main()
            


