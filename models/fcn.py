import torch
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        super(FullyConnectedNet, self).__init__()
        self.layers = nn.ModuleList()

        # Add the input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        for layer_idx in range(num_layers - 2):
            # Add linear layer
            self.layers.append(nn.Linear(hidden_size, hidden_size))

            # Add ReLU activation every two layers
            if (layer_idx + 1) % 2 == 0:
                self.layers.append(nn.ReLU())

            # Add dropout layer every five layers
            if (layer_idx + 1) % 5 == 0:
                self.layers.append(nn.Dropout(dropout_prob))

        # Add the output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Set the dimensions and hyperparameters
input_size = 100
hidden_size = 200
output_size = 10
num_layers = 50
dropout_prob = 0.2

# Create an instance of the network
net = FullyConnectedNet(input_size, hidden_size, output_size, num_layers, dropout_prob)

# Print the network architecture
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Network: {type(net)}, Parameters={num_params/1000000:.2f}M")
