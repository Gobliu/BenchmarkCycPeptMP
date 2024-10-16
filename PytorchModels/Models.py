import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, final_activation):
        super().__init__()
        self.em = nn.Embedding(num_embeddings=27, embedding_dim=128)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(103, hidden_size)
        # self.classifier = nn.Linear(hidden_size, output_size)
        self.final_activation = final_activation
        # self.norm = nn.BatchNorm1d(32)

    def forward(self, x1, x2):

        h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size).to(device)
        x1 = self.em(x1)
        out, _ = self.lstm(x1, (h0, c0))
        out_lstm = out[:, -1, :]
        output = self.classifier(out_lstm)
        output = self.sig(output)

        return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_embeddings, final_activation):
        super().__init__()
        self.em = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.final_activation = final_activation

        self.norm = nn.BatchNorm1d(32)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        x = self.em(x)
        out, _ = self.rnn(x, h0)
        output = self.final_activation(self.fc(out[:, -1, :]))
        return output
