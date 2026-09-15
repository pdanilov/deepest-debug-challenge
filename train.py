import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T

import dataloader
import model

SEED = 0
NUM_HIDDEN_LAYERS = 3
LEARNING_RATE = 1e-3
NUM_TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 10000
LOG_EVERY_STEPS = 50

# the .npz stores raw uint8 pixels in 0..255, but the standard MNIST statistics below
# are defined for the [0, 1] range, so we have to rescale before normalizing
PIXEL_MAX = 255.0
NORMALIZE = T.Normalize((0.1307,), (0.3081,))

# augmentation belongs to training only: evaluation has to stay deterministic
TRAIN_AUGMENT = T.RandomRotation(degrees=10)


def make_collate_fn(augment: bool):
    # builds a collate_fn that rescales, optionally augments, and normalizes a whole batch

    def collate_fn(data: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        # default_collate already stacks the samples into [batch, height, width]
        data_x, data_y = torch.utils.data.default_collate(data)

        # torchvision transforms want a channel dim: [batch, channel, height, width]
        data_x = data_x.unsqueeze(1) / PIXEL_MAX

        if augment:
            # RandomRotation draws one angle per call, so feeding it the whole batch would
            # reuse a single angle for every sample: rotate them one by one instead.
            # this has to happen before NORMALIZE, since rotation fills the exposed
            # corners with 0, which means "black" only while the range is still [0, 1]
            data_x = torch.stack([TRAIN_AUGMENT(sample_x) for sample_x in data_x])

        return NORMALIZE(data_x).squeeze(1), data_y

    return collate_fn


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # this line automatically determines which device to use
    # if you have a fancy NVIDIA GPU the code uses its horsepower. if not, it's fine: the code uses CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################  Data Loading & Augmentation   ##############################
    (x_train, y_train), (x_test, y_test) = dataloader.load_mnist()

    dataset_train = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    mnist_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(augment=True),
    )

    dataset_test = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
    mnist_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=TEST_BATCH_SIZE,
        collate_fn=make_collate_fn(augment=False),
    )

    ##########################################################################################

    ##############################  Neural Networks Definition & Training  ##############################

    # we define the AI using a neural network
    # .to(device) automatically loads the model to the pre-defined device (GPU or CPU)
    neural_net = model.FeedForwardNet(num_hidden_layers=NUM_HIDDEN_LAYERS).to(device)

    # we use an optimizer that trains the AI
    # heard that Adam is good, so use it
    optimizer = optim.Adam(neural_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # now we defined all the necessary things, let's train the AI
    print("\nTraining phase")
    neural_net.train()  # set the model to training mode
    training_step = 0
    acc_loss = 0.0
    n_seen = 0
    for epoch in range(1, NUM_TRAIN_EPOCHS + 1):
        for input_data, target_data in mnist_train:
            # put the input & target data to the auto-defined device (GPU or CPU)
            input_data, target_data = input_data.to(device), target_data.to(device)

            # feed input data to the network
            output = neural_net(input_data)

            # train the model using backpropagation
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_loss += loss.item() * len(target_data)
            n_seen += len(target_data)
            # print the train log every steps
            if (training_step + 1) % LOG_EVERY_STEPS == 0:
                print(f"\tLoss: {acc_loss / n_seen:.3f} [epoch: {epoch}/{NUM_TRAIN_EPOCHS}]", end="\r")
                sys.stdout.flush()
                acc_loss = 0.0
                n_seen = 0
            training_step += 1
        print()

    ##########################################################################################

    ##############################  Evaluation of the Trained Neural Networks   ##############################

    # let's test the trained AI: feed the test data and get the test accuracy
    correct = 0.0
    test_loss = 0.0

    neural_net.eval()  # set the model to evaluation mode
    # pytorch uses no_grad() context manager for evaluation phase: it does not store the history & grads
    # so it's much faster and memory-efficient
    with torch.no_grad():
        for input_data, target_data in mnist_test:
            # same as training phase
            input_data, target_data = input_data.to(device), target_data.to(device)
            output = neural_net(input_data)

            pred_normal = output.max(1, keepdim=True)[1]

            # add up prediction results
            correct += pred_normal.eq(target_data.view_as(pred_normal)).cpu().sum().item()

            # calculate cross entropy loss for target data: same as training
            test_loss += criterion(output, target_data).item() * len(target_data)

    # average out the test results
    test_loss /= len(mnist_test.dataset)
    accuracy = 100.0 * correct / len(mnist_test.dataset)

    # print the test result
    print("\nTest phase")
    print(f"\tTest loss: {test_loss:.3f}")
    print(f"\tAccuracy: {accuracy:.3f}%")


if __name__ == "__main__":
    main()
