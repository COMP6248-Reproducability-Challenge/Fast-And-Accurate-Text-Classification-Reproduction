'''
A PyTorch model with Skimming, rereading, and early stopping.

Use REINFORCE with baseline.
Use discounted rewards for an episode.
'''
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Multinomial, Categorical
from torchtext import datasets
from torchtext import data
import os
import time
import numpy as np 
import random
import argparse

from networks import CNN_LSTM, Policy_C, Policy_N, Policy_S, ValueNetwork
from utils import sample_policy_c, sample_policy_n, sample_policy_s, evaluate, compute_policy_value_losses

desc = '''
A PyTorch model with Skimming, rereading, and early stopping.
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
train, test_data = datasets.IMDB.splits(TEXT, LABEL) # 25,000 training and 25,000 testing data
train_data, valid_data = train.split(split_ratio=0.8) # split training data into 20,000 training and 5,000 vlidation sample

# print(f'Number of training examples: {len(train_data)}')
# print(f'Number of validation examples: {len(valid_data)}')
# print(f'Number of testing examples: {len(test_data)}')

MAX_VOCAB_SIZE = 25_000

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
'''Compute FLOPs(Floating point operations) of the models.
Specify the computational costs for the models above

cnn_cost: CNN model(which is separated from CNN_LSTM above)
s_cost: policy s(stopping module)
c_cost: policy c(classifier)
lstm_cost: LSTM model(which is separated from CNN_LSTM above)
'''
cnn_cost = 1024000
s_cost = 50050
c_cost = 16770 
n_cost = 50310
lstm_cost = 286720
clstm_cost = cnn_cost + lstm_cost
# the number of training epoches
num_of_epoch = 10
# the number of batch size for gradient descent when training
batch_sz = 50

# set up the criterion
criterion = nn.CrossEntropyLoss().to(device)
# set up models
clstm = CNN_LSTM(INPUT_DIM, EMBEDDING_DIM, KER_SIZE, N_FILTERS, HIDDEN_DIM).to(device)
print(clstm)
policy_s = Policy_S(HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
policy_n = Policy_N(HIDDEN_DIM, HIDDEN_DIM, MAX_K).to(device)
policy_c = Policy_C(HIDDEN_DIM, HIDDEN_DIM, LABEL_DIM).to(device)
value_net = ValueNetwork(HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)


# set up optimiser
params_pg = list(policy_s.parameters()) + list(policy_c.parameters()) + list(policy_n.parameters())
optim_loss = optim.Adam(clstm.parameters(), lr=learning_rate)
optim_policy = optim.Adam(params_pg, lr=learning_rate)
optim_value = optim.Adam(value_net.parameters(), lr=learning_rate)

# add pretrained embeddings
pretrained_embeddings = TEXT.vocab.vectors
clstm.embedding.weight.data.copy_(pretrained_embeddings)
clstm.embedding.weight.requires_grad = True  # update the initial weights

# set the default tensor type for GPU
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
        # set the models in training mode
        clstm.train()
        policy_s.train()
        policy_n.train()
        policy_c.train()
        # reset the count of reread_or_skim_times
        reread_or_skim_times = 0
        policy_loss_sum = []
        encoder_loss_sum = []
        baseline_value_batch = []
        for index, train in enumerate(train_iterator):
            label = train.label.to(torch.long)  # for cross entropy loss, the long type is required
            text = train.text.view(CHUNCK_SIZE, BATCH_SIZE, CHUNCK_SIZE) # transform 1*400 to 20*1*20
            curr_step = 0  # the position of the current chunk
            h_0 = torch.zeros([1,1,128]).to(device)  # run on GPU
            count = 0  # maximum skim/reread time: 5
            baseline_value_ep = []
            saved_log_probs = []  # for the use of policy gradient update
            # collect the computational costs for every time step
            cost_ep = []  
            while curr_step < CHUNCK_SIZE and count < 5: 
                # Loop until a text can be classified or currstep is up to 20 or count reach the maximum i.e. 5.
                # update count
                count += 1
                # pass the input through cnn-lstm and policy s
                text_input = text[curr_step] # text_input 1*20
                ht = clstm(text_input, h_0)  # 1 * 128
                # separate the value which is the input of value net
                ht_ = ht.clone().detach().requires_grad_(True)
                # compute a baseline value for the value network
                bi = value_net(ht_)
                # 1 * 1 * 128, next input of lstm
                h_0 = ht.unsqueeze(0)
                # draw a stop decision
                stop_decision, log_prob_s = sample_policy_s(ht, policy_s)
                stop_decision = stop_decision.item()
                if stop_decision == 1: # classify
                    break
                else: 
                    reread_or_skim_times += 1
                    # draw an action (reread or skip)
                    step, log_prob_n = sample_policy_n(ht, policy_n)
                    curr_step += int(step)  # reread or skip
                    if curr_step < CHUNCK_SIZE and count < 5:
                        # If the code can still execute the next loop, it is not the last time step.
                        cost_ep.append(clstm_cost + s_cost + n_cost)
                        # add the baseline value
                        baseline_value_ep.append(bi)
                        # add the log prob for the current actions
                        saved_log_probs.append(log_prob_s + log_prob_n)
            # draw a predicted label
            output_c = policy_c(ht)
            # cross entrpy loss input shape: input(N, C), target(N)
            loss = criterion(output_c, label)  # positive value
            # draw a predicted label 
            pred_label, log_prob_c = sample_policy_c(output_c)
            if stop_decision == 1:
                # add the cost of the last time step
                cost_ep.append(clstm_cost + s_cost + c_cost)
                saved_log_probs.append(log_prob_s + log_prob_c)
            else:
                # add the cost of the last time step
                cost_ep.append(clstm_cost + s_cost + c_cost + n_cost)
                # At the moment, the probability of drawing a stop decision is 1,
                # so its log probability is zero which can be ignored in th sum.
                saved_log_probs.append(log_prob_c.unsqueeze(0))
            # add the baseline value
            baseline_value_ep.append(bi)
            # add the cross entropy loss
            encoder_loss_sum.append(loss)
            # compute the policy losses and value losses for the current episode
            policy_loss_ep, value_losses = compute_policy_value_losses(cost_ep, loss, saved_log_probs, baseline_value_ep, alpha, gamma)
            policy_loss_sum.append(torch.cat(policy_loss_ep).sum())
            baseline_value_batch.append(torch.cat(value_losses).sum())
            # update gradients
            if (index + 1) % batch_sz == 0:  # take the average of 50 samples
                finish_episode(policy_loss_sum, encoder_loss_sum, baseline_value_batch)
                del policy_loss_sum[:], encoder_loss_sum[:], baseline_value_batch[:]
                
            if (index + 1) % 2000 == 0:
                print(f'\n current episode: {index + 1}')
                # log the current position of the text which the agent has gone through
                print('curr_step: ', curr_step)
                # log the sum of the rereading and skimming times
                print(f'current reread_or_skim_times: {reread_or_skim_times}')


        print('Epoch time elapsed: %.2f s' % (time.time() - start))
        print('reread_or_skim_times in this epoch:', reread_or_skim_times)
        count_all, count_correct = evaluate(clstm, policy_s, policy_n, policy_c, valid_iterator)
        print('Epoch: %s, Accuracy on the validation set: %.2f' % (epoch + 1, count_correct / count_all))
        count_all, count_correct = evaluate(clstm, policy_s, policy_n, policy_c, train_iterator)
        print('Epoch: %s, Accuracy on the training set: %.2f' % (epoch + 1, count_correct / count_all))
        
    print('Compute the accuracy on the testing set...')
    count_all, count_correct = evaluate(clstm, policy_s, policy_n, policy_c, test_iterator)
    print('Accuracy on the testing set: %.2f' % (count_correct / count_all))


if __name__ == '__main__':
    main()
