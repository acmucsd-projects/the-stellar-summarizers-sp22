from torch import nn

class Model(nn.Module):
    """Custom model using LSTM that takes a sequence of frames and outputs a importance score"""
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # defining the layers
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, output_size)

    def forward(self, input):
        output, hidden = self.rnn(input)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.out(output)

        return output, hidden

# test code
if __name__ == '__main__':
    feature_size = 20
    model = Model(feature_size, 1, 512, 4)
    print(model)