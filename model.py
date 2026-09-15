import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 3,
        in_size: int = 784,
        dropout_p: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_size = in_size
        self.feed_forward = nn.Sequential()
        for _ in range(num_hidden_layers):
            self.feed_forward.extend([nn.Linear(self.in_size, self.in_size), nn.Dropout(p=dropout_p), nn.ReLU()])
        self.feed_forward.append(nn.Linear(self.in_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x.view(-1, self.in_size))
