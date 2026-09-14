import torch.nn.functional as F
from torch import nn

# WARNING: this code is full of (ML-logical) bugs. can you squash them all?


class FeedForwardNet(nn.Module):
    # a standard feed-forward neural network
    # consists of brain layers with fc_last as final output layer
    # fc_last outputs 10-dimensional vector, corresponding to scores for digits

    def __init__(self, num_layers):
        # this is a constructor for superclass in pytorch
        super(FeedForwardNet, self).__init__()

        # num_layers determines the number of brain layers to use
        # since it is "deep learning", we have to go deeper
        self.num_layers = num_layers

        # these lines define parameters for the model
        self.fc_layers = nn.ModuleList([])
        self.fc_layers.extend([nn.Linear(784, 784) for i in range(self.num_layers)])
        self.fc_last = nn.Linear(784, 10)

        # initialize weights
        for i in range(len(self.fc_layers)):
            self.fc_layers[i].weight.data.fill_(0)
            self.fc_layers[i].bias.data.fill_(0)
        self.fc_last.weight.data.fill_(0)
        self.fc_last.bias.data.fill_(0)

    def forward(self, x):
        # defines forward pass of the model
        # takes x as input, computes output with the model parameters

        # flatten the 28x28 images to a 1-dimensional (784) vector
        x = x.view(-1, 784)

        # this is the forward pass of the model: apply the weight, non-linear activation, and dropout
        for i in range(self.num_layers):
            fc_feature = F.dropout(F.sigmoid(self.fc_layers[i](x)), p=0.99)

        # apply the final layer to convert deep features to digit scores
        fc_last = F.dropout(F.sigmoid(self.fc_last(fc_feature)), p=0.99)

        return fc_last
