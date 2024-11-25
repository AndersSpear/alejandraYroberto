# Alex Jian Zheng
from pathlib import Path
import pyshark
import torch
import torch.nn as nn
import numpy as np
from numpy import zeros
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import argparse

torch.manual_seed(1701)

vocab = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2),
 (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2),
 (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2),
 (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2),
 (0, 2, 2, 0), (0, 2, 2, 1), (0, 2, 2, 2), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2),
 (1, 0, 1, 0), (1, 0, 1, 1), (1, 0, 1, 2), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2),
 (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 2),
 (1, 1, 2, 0), (1, 1, 2, 1), (1, 1, 2, 2), (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2),
 (1, 2, 1, 0), (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 2, 0), (1, 2, 2, 1), (1, 2, 2, 2),
 (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 0, 2), (2, 0, 1, 0), (2, 0, 1, 1), (2, 0, 1, 2),
 (2, 0, 2, 0), (2, 0, 2, 1), (2, 0, 2, 2), (2, 1, 0, 0), (2, 1, 0, 1), (2, 1, 0, 2),
 (2, 1, 1, 0), (2, 1, 1, 1), (2, 1, 1, 2), (2, 1, 2, 0), (2, 1, 2, 1), (2, 1, 2, 2),
 (2, 2, 0, 0), (2, 2, 0, 1), (2, 2, 0, 2), (2, 2, 1, 0), (2, 2, 1, 1), (2, 2, 1, 2),
 (2, 2, 2, 0), (2, 2, 2, 1), (2, 2, 2, 2)]


class LangDataset(Dataset):
    def __init__(self, data):
        self.n_samples, self.n_features = data.shape
        # The first column is label, the rest are the features
        self.n_features -= 1

        assert (self.n_samples, self.n_features) == (240, 81)
        self.feature = torch.from_numpy(data[:, 1:].astype(np.float32)) # size [n_samples, n_features]
        self.label = torch.from_numpy(data[:, [0]].astype(np.float32)) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def pcap_to_lengths(pcap_file) -> [int]:
    length = []
    capture = pyshark.FileCapture(pcap_file)
    for packet in capture:
        length.append(packet.length)

    return length

def lengths_to_tokens(lengths: [int]) -> [int]:
    # 0 = 2 smallest
    # 2 = largest length
    # 1 = everything else

    lengths_mod = lengths
    lengths_mod.remove(min(lengths_mod))

    min_length2 = min(lengths_mod) # second smallest
    min_length = min(lengths)      # smallest
    max_length = max(lengths)      # largest

    for i in range(len(lengths)):
        if lengths[i] == min_length or lengths[i] == min_length2:
            lengths[i] = 0
        elif lengths[i] == max_length:
            lengths[i] = 2
        else:
            lengths[i] = 1

    return lengths

def tokens_to_tuples(tokens: [int]) -> [(int, int, int, int)]:
    tuples = []
    i = 0
    while i + 3 < len(tokens):
        tuples.append((tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]))
        i = i+1

    return tuples


def count_tuples(words: [(int, int, int, int)]) -> [int]:
    # return type is {int: int} where key is index of tuple in vocab and value is the count
    count = []
    for i in range(len(vocab)):
        count.append(words.count(vocab[i]))

    return count


def read_dataset(lang1, lang2):
    """
    :param lang1: directory names of pcap files of lang1 (string)
    :param lang2: directory names of pcap files of lang2 (string)

    read in files and create a list of pcap files of each language

    turn each pcap file into a list of individual packet lengths
    identify the lengths that each token represents
    turn each list of packet lengths into a list of tokens
    create a list of 4-tuples of tokens in each pcap file.

    create lengths1 and lengths2
    length1: list of lists of lengths

    tokens =
    [
    0 token representing two smallest lengths,
    1 token representing middle lengths,
    2 token representing largest length
    ]

    vocab =
    [
    (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1), ...]

    [m, s, s, s, m, l, m, s, m, m]
    (m, s, s, s), (s, s, s, m), (s, s, m, l), (s, m, l, m)

    matrix will be a 2D array where each row is a sample file and columns are class(language), the 81 columns
    of the count in vocab

    """
    directory1 = Path(lang1)
    directory2 = Path(lang2)

    matrix = [] # 2D array

    # read in every .pcap file in directories lang1 and lang2 into a list

    pcap_files1 = list(directory1.glob('*.pcapng'))
    pcap_files2 = list(directory2.glob('*.pcapng'))

    # convert pcaps to lengths
    lang1_lengths = [pcap_to_lengths(f) for f in pcap_files1]
    lang2_lengths = [pcap_to_lengths(f) for f in pcap_files2]

    # convert lengths to tokens
    lang1_tokens = [lengths_to_tokens(l) for l in lang1_lengths]
    lang2_tokens = [lengths_to_tokens(l) for l in lang2_lengths]

    # convert tokens to tuples
    lang1_tuples = [tokens_to_tuples(l) for l in lang1_tokens]
    lang2_tuples = [tokens_to_tuples(l) for l in lang2_tokens]

    # convert tuples to counts of each sample to insert into matrix -> [{int: int}]
    lang1_counts = [count_tuples(t) for t in lang1_tuples]
    lang2_counts = [count_tuples(t) for t in lang2_tuples]

    final = zeros((len(lang1_counts) + len(lang2_counts), (len(vocab) + 1)))

    final[:len(lang1_counts), 1:] = np.asmatrix(lang1_counts)
    final[:len(lang1_counts), 1] = 0

    final[len(lang1_counts):, 1:] = lang2_counts
    final[len(lang1_counts):, 1] = 1

    lang1_m = np.asmatrix(lang1_counts)
    lang2_m = np.asmatrix(lang2_counts)


    # create the matrix
    for i in range(len(vocab)):
        sample_count1 = [0] * (len(vocab) + 1)
        sample_count2 = [0] * (len(vocab) + 1)
        # class = 0 for lang1, class = 1 for lang2
        sample_count2[0] = 1

        sample_count1[i + 1] = lang1_counts[i]
        sample_count2[i + 1] = lang2_counts[i]

        # add both arrays to the matrix
        matrix.append(sample_count1)
        matrix.append(sample_count2)

    return matrix

class SimpleLogreg(nn.Module):
    def __init__(self, num_features):
        """
        Initialize the parameters you'll need for the model.

        :param num_features: The number of features in the linear model
        """
        super(SimpleLogreg, self).__init__()
        self.linear = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.linear(x)

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
    function so we could write tests.


    :param epoch: The current epoch
    :param ex: Which example / minibatch you're one
    :param model: The model you're optimizing
    :param inputs: The current set of inputs
    :param labels: The labels for those inputs

    A) get predictions
    B) compute the loss from that prediction
    C) backprop
    D) update the parameters
    """
    optimizer.zero_grad()
    prediction = model(inputs)
    loss = criterion(prediction, labels)
    loss.backward()
    optimizer.step()

    if (ex+1) % 20 == 0:
      acc_train = model.evaluate(train)
      acc_test = model.evaluate(test)
      print(f'Epoch: {epoch+1}/{num_epochs}, Example {ex}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f} test_acc = {acc_test.item():.4f}')



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    #''' Switch between the toy and REAL EXAMPLES
    argparser.add_argument("--lang1", help="Language 1 class",
                           type=str, default="./data/smallenglish")
    argparser.add_argument("--lang2", help="Language 2 class",
                           type=str, default="./data/smallspanish")
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=5)
    argparser.add_argument("--batch", help="Number of items in each batch",
                           type=int, default=1)
    argparser.add_argument("--learnrate", help="Learning rate for SGD",
                           type=float, default=0.1)

    args = argparser.parse_args()

    langdata = read_dataset(args.lang1, args.lang2)
    """
    each row is the count of each 
    first column is the language type
    
    """

    train_np, test_np = train_test_split(langdata, test_size=0.15, random_state=1234)
    train, test = LangDataset(train_np), LangDataset(test_np)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    logreg = SimpleLogreg(train.n_features)

    num_epochs = args.passes
    batch = args.batch
    total_samples = len(train)

    # Replace these with the correct loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(logreg.parameters(), lr=args.learnrate)
    
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
