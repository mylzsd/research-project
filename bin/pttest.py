import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier

import pandas as pd
import reader as rdr


class DQN(nn.Module):
    def __init__(self, input_size, hiddens, output_size):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        for hidden in hiddens:
            self.layers.append(nn.Linear(input_size, hidden))
            input_size = hidden
        self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def main():
    dataset = 'human_activity'
    data = rdr.read(dataset)
    train, test = rdr.splitByPortion(data, 0.8, 6666)

    label_map = dict()
    for l in train.iloc[:, -1]:
        if l not in label_map:
            label_map[l] = len(label_map)
    for l in test.iloc[:, -1]:
        if l not in label_map:
            label_map[l] = len(label_map)
    num_labels = len(label_map)

    x_train = train.iloc[:, :-1].astype(float)
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = []
    for l in train.iloc[:, -1]:
        y_temp = [0.0] * num_labels
        y_temp[label_map[l]] = 1.0
        y_train.append(y_temp)
    # y_train = pd.DataFrame(y_train)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_test = test.iloc[:, :-1].astype(float)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_test = []
    for l in test.iloc[:, -1]:
        y_temp = [0.0] * num_labels
        y_temp[label_map[l]] = 1.0
        y_test.append(y_temp)
    # y_test = pd.DataFrame(y_test)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    hiddens = (128, 128, 16)

    model = DQN(train.shape[1] - 1, hiddens, num_labels)
    # print(list(model.parameters()))
    # output = model(x_train)
    # print(output.shape)
    # print(output)
    criterion = nn.MSELoss()
    # loss = criterion(output, output)
    # print(loss)
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    p_loss = -1
    for i in range(20000):
        optimizer.zero_grad()
        out_train = model(x_train)
        loss = criterion(out_train, y_train)
        loss.backward()
        optimizer.step()
        if p_loss != -1:
            d_loss = p_loss - loss
            if d_loss < 0.00001:
                print('stop at epoch %d due to convergence' % (i + 1))
                break
        p_loss = loss
        if (i + 1) % 2000 == 0:
            out_test = model(x_test)
            correct = torch.eq(torch.argmax(out_test, dim=1), torch.argmax(y_test, dim=1))
            test_accu = torch.mean(correct.float())
            print('epoch: %d, accuracy: %.5f, loss: %.5f' % (i + 1, test_accu.detach().numpy(), loss.detach().numpy()))
    out_test = model(x_test)
    correct = torch.eq(torch.argmax(out_test, dim=1), torch.argmax(y_test, dim=1))
    test_accu = torch.mean(correct.float())
    pt_score = test_accu

    sl_nn = MLPClassifier(hidden_layer_sizes=hiddens, activation='relu', solver='adam', max_iter=20000, random_state=6666)
    sl_nn.fit(x_train, y_train)
    sl_score = sl_nn.score(x_test, y_test)

    print('dataset: %s\ntraining: %s\ntesting: %s\n# labels: %d'
            % (dataset, str(train.shape), str(test.shape), num_labels))
    print('pt: %f\nsl: %f' % (pt_score, sl_score))


if __name__ == '__main__':
    main()



