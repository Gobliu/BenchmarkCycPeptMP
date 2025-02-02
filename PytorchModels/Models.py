import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_embeddings, final_activation):
        super().__init__()
        # padding_idx is important, otherwise predictions tend to be the same
        self.em = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.final_activation = final_activation
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = self.em(x)
        # x = self.layer_norm(x)
        out, out_ = self.rnn(x)
        # print(out.shape, out_.shape)
        output = self.final_activation(self.fc(out[:, -1, :]))
        # print('output', output.shape)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_embeddings, final_activation):
        super().__init__()
        self.em = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.final_activation = final_activation
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Embed and optionally normalize (consider removing LayerNorm if not necessary)
        x = self.em(x)
        # x = self.layer_norm(x)  # Consider disabling for testing purposes

        # LSTM forward pass
        out, _ = self.lstm(x)
        # print(out.shape)  # Fixed typo

        # Fully connected layer and activation
        output = self.final_activation(self.fc(out[:, -1, :]))
        # print('output', output.shape, output)

        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_embeddings, final_activation):
        super(GRU, self).__init__()
        self.em = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.final_activation = final_activation

    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = self.em(x)

        # Forward propagate through GRU
        out, _ = self.gru(x)

        # Pass the last hidden state through the fully connected layer
        output = self.final_activation(self.fc(out[:, -1, :]))
        return output
