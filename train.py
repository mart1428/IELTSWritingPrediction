import torch
import torch.optim
import torch.nn as nn

import time

def get_model_name(name, batch_size, lr, epoch):
    return '{0}_bs{1}_lr{2}_epoch{3}'.format(name, batch_size, lr, epoch)

def trainIELTSScorer(model, train_loader, val_loader, batch_size, lr = 0.001, epochs = 30):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay= 0.000001)
    criterion = torch.nn.CrossEntropyLoss(torch.Tensor([0.6,0.4]))
    criterion.to(device)

    print(f'Training in {device}')
    start_time = time.time()
    for e in range(epochs):
        model.unbatched = False
        running_loss = 0
        running_error = 0
        total = 0
        corr = 0
        model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # print(labels)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            running_error += (predicted != labels).long().sum()
            corr += (predicted == labels).long().sum()
            total += len(labels)


        train_loss = running_loss/len(train_loader)
        train_error = running_error/len(train_loader.dataset)
        train_accuracy = corr/total

        model.eval()
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

        print(f'Epoch {e+1} - Train| Loss: {train_loss:.3f}, Error: {train_error:.3f}, Acc: {train_accuracy:.2%} ||| Val| Loss:{val_loss:.3f}, Error: {val_error:.3f}, Acc: {val_accuracy:.2%}')
        end_time = time.time()-start_time
        print(f'Time after epoch {e+1}: {end_time:.2f}s')

    torch.save(model.state_dict(), get_model_name(model.name, batch_size, lr, e+1))
