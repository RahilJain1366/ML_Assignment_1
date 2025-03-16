import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        batch_size = inputs.size(1)
        hidden = torch.zeros(self.numOfLayer, batch_size, self.h)
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, hidden_layer = self.rnn(inputs,hidden)
        # [to fill] obtain output layer representations
        output = self.W(hidden_layer[-1])
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(input_dim=50, h = args.hidden_dim)  # Fill in parameters
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    word_embedding = pickle.load(open('./Data_Embedding/word_embedding.pkl', 'rb'))

    if unk not in word_embedding:
        word_embedding[unk] = np.zeros(50)

    stopping_condition = False
    epoch = 0
    training_accuracies = []
    validation_accuracies = []
    training_losses = []
    last_train_accuracy = 0
    last_validation_accuracy = 0

    error_distribution = []
    train_predicted_labels = []
    train_actual_labels = []
    val_predicted_labels = []
    val_actual_labels = []

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(0, N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            # Collect inputs and labels
            example_loss = 0
            for example_index in range(minibatch_size):
                if minibatch_index + example_index >= len(train_data):
                    break
                input_words, gold_label = train_data[minibatch_index + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                #vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]
                vectors = [word_embedding.get(word.lower(), word_embedding[unk]) for word in input_words]
                
                # Transform the input into required shape
                vectors = torch.tensor(vectors, dtype=torch.float32).unsqueeze(1)
                
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output, torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                    # print(predicted_label, gold_label)
                train_predicted_labels.append(predicted_label.item())
                train_actual_labels.append(gold_label)

            example_loss.backward()
            optimizer.step()

            loss_total += example_loss.item()
            loss_count += 1
            
        training_loss = loss_total/loss_count
        trainning_accuracy = correct/total
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, trainning_accuracy))
        training_losses.append(training_loss)

        model.eval()
        correct = 0
        total = 0
        print("Validation started for epoch {}".format(epoch + 1))
        with torch.no_grad():
            for input_words, label in tqdm(valid_data):
                input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                input_vectors = [word_embedding.get(word.lower(), word_embedding[unk]) for word in input_words]
                input_tensor = torch.tensor(input_vectors, dtype=torch.float32).unsqueeze(1)

                output = model(input_tensor)
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == label)
                total += 1

                val_predicted_labels.append(predicted_label.item())
                val_actual_labels.append(label)

                if predicted_label != label:
                    error_distribution.append({
                        "input": " ".join(input_words),
                        "predicted": predicted_label.item(),
                        "actual": label
                    })

        val_accuracy = correct / total
        validation_accuracies.append(val_accuracy)
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        

        if val_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = val_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1

   
    # You may find it beneficial to keep track of training accuracy or training loss;
    results = {
    "train_losses": training_losses,  
    "val_accuracies": validation_accuracies
    }

    os.makedirs("RNN_results", exist_ok=True)
    with open("RNN_results/RNN_results.json", "w") as f:
        json.dump(results, f)

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
    with open('RNN_results/error_distribution.json', 'w') as f:
        json.dump(error_distribution, f)
    
    with open('RNN_results/train_predictions.json', 'w') as f:
        json.dump({'predicted': train_predicted_labels, 'actual': train_actual_labels}, f)

    with open('RNN_results/val_predictions.json', 'w') as f:
        json.dump({'predicted': val_predicted_labels, 'actual': val_actual_labels}, f)

    # To Plot learning curve
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), training_losses, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.plot(range(1, epoch + 1), validation_accuracies, marker='o', linestyle='-', color='g', label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy / Loss")
    plt.title("Training Loss per Epoch")
    plt.title("Validation Accuracy per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("RNN_results/learning_curve.png")
    plt.show()
    