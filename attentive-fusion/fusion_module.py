import numpy as np
import torch
import torch.nn as nn

rv_emb_path = ""
sv_emb_path = ""
POI_emb_path = ""


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b = nn.Parameter(torch.Tensor(input_size))
        self.u = nn.Parameter(torch.Tensor(input_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)
        nn.init.xavier_uniform_(self.u)

    def forward(self, input_rv, input_sv):
        u_rv = torch.tanh(torch.matmul(input_rv, self.W) + self.b)
        u_sv = torch.tanh(torch.matmul(input_sv, self.W) + self.b)
        rv_attention_weights = torch.matmul(u_rv, self.u)
        sv_attention_weights = torch.matmul(u_sv, self.u)
        attention_weights = torch.softmax(torch.cat((rv_attention_weights, sv_attention_weights)), dim=0)
        output = input_rv * attention_weights[0] + input_sv * attention_weights[1]
        return output


rv_emb = torch.from_numpy(np.load(rv_emb_path).reshape(746, 128))
sv_emb = torch.from_numpy(np.load(sv_emb_path).reshape(746, 128)).to(torch.float32)
POI_emb = torch.from_numpy(np.load(POI_emb_path).reshape(746, 128)).to(torch.float32)

model = Attention(128)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(len(rv_emb)):
        d = model(rv_emb[i], sv_emb[i])
        loss = loss_fn(d, POI_emb[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(rv_emb)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

final = []
for i in range(len(rv_emb)):
    d = model(rv_emb[i], sv_emb[i]).detach().numpy().tolist()
    print(d)
    final.append(d)

np.save("", np.array(final))
