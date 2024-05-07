import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from fusion import emb_fusion

torch.manual_seed(1)
np.random.seed(1)


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_num = len(hidden_dim)
        if self.hidden_num == 1:
            self.fc1 = nn.Linear(input_dim, hidden_dim[0])
            self.fc2 = nn.Linear(hidden_dim[0], output_dim)
        if self.hidden_num == 2:
            self.fc1 = nn.Linear(input_dim, hidden_dim[0])
            self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
            self.fc3 = nn.Linear(hidden_dim[1], output_dim)
        if self.hidden_num == 3:
            self.fc1 = nn.Linear(input_dim, hidden_dim[0])
            self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
            self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
            self.fc4 = nn.Linear(hidden_dim[2], output_dim)
        if self.hidden_num == 4:
            self.fc1 = nn.Linear(input_dim, hidden_dim[0])
            self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
            self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
            self.fc4 = nn.Linear(hidden_dim[2], hidden_dim[3])
            self.fc5 = nn.Linear(hidden_dim[3], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.hidden_num == 1:
            return self.fc2(self.relu(self.fc1(x)))
        if self.hidden_num == 2:
            return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        if self.hidden_num == 3:
            return self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))
        if self.hidden_num == 4:
            return self.fc5(self.relu(self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))))


def train(model, dataloader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(dataloader.dataset)
    return train_loss


def test(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.cpu().numpy())
    test_loss /= len(dataloader.dataset)
    return test_loss, np.array(predictions)


rv_features = np.load("").reshape(517, 128)
sv_features = np.load("").reshape(517, 128)
POI_features = np.load("").reshape(517, 128)

features = emb_fusion(rv_features, sv_features, POI_features, 'fusion')
indicators = np.load("")


X_train, X_test, y_train, y_test = train_test_split(features, indicators, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2 / 0.8)


train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = MyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = MyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


input_dim = X_train.shape[1]
hidden_dim = [512, 256, 128, 64]
output_dim = 1
mlp = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.0005)


best_val_loss = np.inf
early_stop_count = 0
train_loss_line = []
val_loss_line = []
for epoch in range(20000):
    train_loss = train(mlp, train_loader, criterion, optimizer)
    val_loss, val_pred = test(mlp, val_loader, criterion)
    train_loss_line.append(train_loss)
    val_loss_line.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(mlp.state_dict(), "best_model.pth")
        early_stop_count = 0
    else:
        early_stop_count = early_stop_count + 1
        if early_stop_count > 50:
            print("Early Stop in Epoch {}".format(epoch))
            break
    if epoch % 50 == 0:
        r2_val = r2_score(y_val, val_pred)
        print("Epoch {} - Train loss: {:.4f}, Validation loss: {:.4f}, R2: {:.4f}".format(epoch, train_loss, val_loss, r2_val))


mlp.load_state_dict(torch.load("best_model.pth"))
test_loss, test_pred = test(mlp, test_loader, criterion)

rmse = np.sqrt(mean_squared_error(y_test, test_pred))
r2 = r2_score(y_test, test_pred)
mape = np.mean(np.abs((y_test - test_pred) / y_test))
print("Test Loss: {:.4f}, Test RMSE: {:.4f}, Test R2: {:.4f}, Test MAPE: {:.4f}".format(test_loss, rmse, r2, mape))


np.save("", np.array(y_test))
np.save("", np.array(test_pred))

train_loss, train_pred = test(mlp, val_loader, criterion)
r2_train = r2_score(y_val, train_pred)
print(r2_train)
np.save("", np.array(y_val))
np.save("", np.array(train_pred))
