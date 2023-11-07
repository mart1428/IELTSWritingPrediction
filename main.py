import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchtext
from torchtext.vocab import GloVe

from data import retrieve_data, tokenize_data, get_data_loader
from model import IELTSScorer
from train import trainIELTSScorer


def trainValTestModel(model, train_loader, val_loader, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.6,0.4]))
    criterion.to(device)
    print(f'Testing in {device}')
    model.eval()

    with torch.no_grad():
        running_loss = 0
        running_error = 0
        total = 0
        corr = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            running_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            running_error += (predicted != labels).long().sum()
            corr += (predicted == labels).long().sum()
            total += len(labels)
            

        train_loss = running_loss/len(train_loader)
        train_error = running_error/len(train_loader.dataset)
        train_accuracy = corr/total
    
    with torch.no_grad():
        running_loss = 0
        running_error = 0
        total = 0
        corr = 0

        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            running_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            running_error += (predicted != labels).long().sum()
            corr += (predicted == labels).long().sum()
            total += len(labels)
            

        val_loss = running_loss/len(val_loader)
        val_error = running_error/len(val_loader.dataset)
        val_accuracy = corr/total
        

    with torch.no_grad():
        running_loss = 0
        running_error = 0
        total = 0
        corr = 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            running_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            running_error += (predicted != labels).long().sum()
            corr += (predicted == labels).long().sum()
            total += len(labels)
            

        test_loss = running_loss/len(test_loader)
        test_error = running_error/len(test_loader.dataset)
        test_accuracy = corr/total

    print(f'Train - Loss: {train_loss:.3f}, Error: {train_error:.3f}, Acc: {train_accuracy:.2%} | Val - Loss:{val_loss:.3f}, Error: {val_error:.3f}, Acc: {val_accuracy:.2%} | Test - Loss:{test_loss:.3f}, Error: {test_error:.3f}, Acc: {test_accuracy:.2%}')


if __name__ == '__main__':
    torch.random.manual_seed(1000)

    batch_size = 312
    lr = 0.001


    data = retrieve_data()
    tokenized_data = tokenize_data(data)
    train_loader, val_loader, test_loader = get_data_loader(tokenized_data, batch_size)

    model = IELTSScorer(2)

    loadModel = 1
    if loadModel:
        model_path = 'IELTSScorer_bs312_lr0.001_epoch66'

        model.load_state_dict(torch.load(model_path))

    # trainIELTSScorer(model, train_loader, val_loader, batch_size, lr, 66)

    trainValTestModel(model, train_loader, val_loader, test_loader)

