# Alex Jian Zheng
import random
import math
import torch
import torch.nn as nn
import numpy as np
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import argparse

torch.manual_seed(1701)

class SportsDataset(Dataset):
    def __init__(self, data):
        self.n_samples, self.n_features = data.shape
        # The first column is label, the rest are the features
        self.n_features -= 1
        self.feature = torch.from_numpy(data[:, 1:].astype(np.float32)) # size [n_samples, n_features]
        self.label = torch.from_numpy(data[:, [0]].astype(np.float32)) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

# def read_vocab(vocab_file):
#     vocab = [str(x).split("\t")[0] for x in vocab_file.readlines() if '\t' in x]
#     print("Loaded vocab with %i terms" % len(vocab))
#     return vocab

def pcap_to_lengths(pcap_file) -> [int]:
    exit()

def lengths_to_tokens(lengths: [int]) -> [int]:
    exit()

def tokens_to_tuples(tokens: [int]) -> [(int, int, int, int)]:
    exit()

def count_tuples(words: [(int, int, int, int)],
                 vocab: {(int, int, int, int): int}) -> {int: int}:
    exit()



def read_dataset(lang1, lang2):
    """
    Create a pytorch SportsDataset for the train and test data.

    :param lang1: list of pcap files of lang1
    :param negative: list of pcap files of lang2

    turn each pcap file into a list of individual packet lengths
    identify the lengths that each token represents
    turn each list of packet lengths into a list of tokens
    create a list of 4-tuples of tokens in each pcap file.



    create lengths1 and lengths2
    length1: list of lists of lengths

    tokens =
    [
    s token representing two smallest lengths,
    m token representing middle lengths,
    l token representing largest length
    ]

    vocab =
    [
    (s, s, s, s), (s, s, s, m), (s, s, m, m),
]




    [m, s, s, s, m, l, m, s, m, m]
    (m, s, s, s), (s, s, s, m), (s, s, m, l), (s, m, l, m)



    :param vocab: Vocabulary words file pointer
    """

    # You'll need to change the below line, since you can't assume
    # that you have five documents in your dataset.
    num_docs = 5
    num_docs = len(positive.readlines()) + len(negative.readlines())       
    positive.seek(0)                                                       
    negative.seek(0)

    sportsdata = zeros((num_docs, len(vocab)))
    print("Created datasets with %i rows and some columns" % (num_docs)


    return sportsdata

class SimpleLogreg(nn.Module):
    def __init__(self, num_features):
        """
        Initialize the parameters you'll need for the model.

        :param num_features: The number of features in the linear model
        """
        super(SimpleLogreg, self).__init__()
        # Replace this with a real nn.Module
        self.linear = None

    def forward(self, x):
        """
        Compute the model prediction for an example.

        :param x: Example to evaluate
        """
        return 0.5

    def evaluate(self, data):
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc

    def inspect(self, vocab, limit=10):
        """
        A fundtion to find the top features and print them.
        """

        None
        weights = logreg.linear.weight[0].detach().numpy()


def step(epoch, ex, model, optimizer, criterion, inputs, labels):
    """Take a single step of the optimizer, we factored it into a single
    function so we could write tests.  You should: A) get predictions B)
    compute the loss from that prediction C) backprop D) update the
    parameters

    There's additional code to print updates (for good software
    engineering practices, this should probably be logging, but printing
    is good enough for a homework).

    :param epoch: The current epoch
    :param ex: Which example / minibatch you're one
    :param model: The model you're optimizing
    :param inputs: The current set of inputs
    :param labels: The labels for those inputs
    """

    if (ex+1) % 20 == 0:
      acc_train = model.evaluate(train)
      acc_test = model.evaluate(test)
      print(f'Epoch: {epoch+1}/{num_epochs}, Example {ex}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f} test_acc = {acc_test.item():.4f}')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    #''' Switch between the toy and REAL EXAMPLES
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../logreg/data/positive")
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../logreg/data/negative")
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../logreg/data/vocab")
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=5)
    argparser.add_argument("--batch", help="Number of items in each batch",
                           type=int, default=1)
    argparser.add_argument("--learnrate", help="Learning rate for SGD",
                           type=float, default=0.1)

    args = argparser.parse_args()

    sportsdata = read_dataset(open(args.positive), open(args.negative), open(args.vocab))
    """
    each row is the count of each 
    first column is the language type
    
    """
    sportsdata = np.array([
      [0, 0, 3, 5, 123, 6, 765, 987, 23, 34, 654, 76],
      [0, 5, 32, 12, 4, 11, 76, 8, 43, 2, 6, 1],
      [1, 12, 3, 5, 123, 6, 75, 87, 23, 34, 64, 76],
      [1, 69, 32, 112, 544, 12, 76, 9, 43, 2, 6, 1221],
    ])

    train_np, test_np = train_test_split(sportsdata, test_size=0.15, random_state=1234)
    train, test = SportsDataset(train_np), SportsDataset(test_np)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    logreg = SimpleLogreg(train.n_features)

    num_epochs = args.passes
    batch = args.batch
    total_samples = len(train)

    # Replace these with the correct loss and optimizer
    criterion = None
    optimizer = None
    
    train_loader = DataLoader(dataset=train,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)
    dataiter = iter(train_loader)

    # Iterations
    for epoch in range(num_epochs):
      for ex, (inputs, labels) in enumerate(train_loader):
        # Run your training process
        step(epoch, ex, logreg, optimizer, criterion, inputs, labels)

    # Print out the best features
    vocab = read_vocab(open(args.vocab))
    logreg.inspect(vocab)
