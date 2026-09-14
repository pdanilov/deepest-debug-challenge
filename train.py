import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import dataloader
import model

# WARNING: this code is full of (ML-logical) bugs. can you squash them all?
# we've created a super-awesome AI that recognizes digits!
# but it does not work as expected.. why?

# this line automatically determines which device to use
# if you have a fancy NVIDIA GPU the code uses its horsepower. if not, it's fine: the code uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################  Data Loading & Augmentation   ##############################

# we use MNIST dataset, which contains images of hand-written digits
# or does it?
(x_train, y_train), (x_test, y_test) = dataloader.load_mnist()

# heard that we can enlarge the dataset by augmenting the images
# thankfully, torchvision has built-in method with compose API
transformer = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
        transforms.RandomRotation(degrees=90),
        transforms.RandomResizedCrop(size=28, scale=(0.2, 2.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
)


# pytorch DataLoader can additionally use collate_fn for post-processing the loaded data
# use will define collate_fn to augment the data on-the-fly
def collate_fn(data):
    data_x = []
    data_y = []
    for i, sample in enumerate(data):
        sample_x, sample_y = sample[0], sample[1]
        # unsqueeze the first dim of sample_x since torchvision transform only accepts [channel, height, width]
        # https://github.com/pytorch/vision/issues/408
        sample_x = sample_x.unsqueeze(0)
        sample_x = transformer(sample_x)
        sample_x = sample_x.squeeze(0)
        sample_x *= 255
        data_x.append(sample_x)
        data_y.append(sample_y)
    data_x = torch.stack(data_x)
    data_y = torch.stack(data_y)
    return data_x, data_y


# define the train & test pytorch DataLoader with transformer & collate_fn
dataset_train = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
mnist_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=collate_fn)

dataset_test = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
mnist_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000, collate_fn=collate_fn)

##########################################################################################


##############################  Neural Networks Definition & Training  ##############################

# we define the AI using a neural network
# .to(device) automatically loads the model to the pre-defined device (GPU or CPU)
neural_net = model.FeedForwardNet(num_layers=10).to(device)

# we use an optimizer that trains the AI
# heard that Adam is good, so use it
optimizer = optim.Adam(neural_net.parameters(), lr=1)

# now we defined all the necessary things, let's train the AI
print("\n" + "training phase")
for batch_idx, (input_data, target_data) in enumerate(mnist_train):
    # pytorch needs to "zero-fill" gradients for each train step
    # otherwise, the model adds up the gradients: not what you would expect
    neural_net.zero_grad()

    # put the input & target data to the auto-defined device (GPU or CPU)
    input_data, target_data = input_data.to(device), target_data.to(device)

    # feed input data to the network
    output = neural_net(input_data)

    # apply softmax to convert output to digit scores
    output = F.softmax(output, dim=0)

    # we define how well the model performed by comparing the output to target data
    # cross entropy is a natural choice
    # first, convert the target data to one-hot
    target_data_onehot = np.zeros((target_data.size()[0], 10), dtype=np.float32)
    target_data_onehot[np.arange(target_data.size()[0]), target_data.cpu().data.numpy()] = 1
    target_data_onehot = torch.FloatTensor(torch.from_numpy(target_data_onehot)).to(device)

    # then, calculate the cross entropy error
    loss = -torch.mean(torch.sum(torch.mul(target_data_onehot, torch.log(output)), dim=0))

    # train the model using backpropagation
    loss.backward()
    optimizer.step()

    # print the train log every steps
    if batch_idx % 1 == 0:
        train_log = "Loss: {:.6f}\tTrain: [{}/{} ({:.0f}%)]".format(
            loss.item(), batch_idx * len(input_data), len(mnist_train.dataset), 100.0 * batch_idx / len(mnist_train)
        )
        print(train_log, end="\r")
        sys.stdout.flush()

##########################################################################################


##############################  Evaluation of the Trained Neural Networks   ##############################

# let's test the trained AI: feed the test data and get the test accuracy
correct = 0.0
test_loss = 0.0

# pytorch uses no_grad() context manager for evaluation phase: it does not store the history & grads
# so it's much faster and memory-efficient
with torch.no_grad():
    for batch_idx, (input_data, target_data) in enumerate(mnist_test):
        # same as training phase
        input_data, target_data = input_data.to(device), target_data.to(device)
        output = neural_net(input_data)
        output = F.softmax(output, dim=0)

        # get the index of the max log-probability
        pred_normal = output.data.max(1, keepdim=True)[1]

        # add up prediction results
        correct += pred_normal.eq(target_data.data.view_as(pred_normal)).cpu().sum()

        # calculate cross entropy loss for target data: same as training
        target_data_onehot = np.zeros((target_data.size()[0], 10), dtype=np.float32)
        target_data_onehot[np.arange(target_data.size()[0]), target_data.cpu().data.numpy()] = 1
        target_data_onehot = Variable(torch.FloatTensor(torch.from_numpy(target_data_onehot))).to(device)
        test_loss += -torch.sum(torch.sum(torch.mul(target_data_onehot, torch.log(output)), dim=0))

# average out the test results
test_loss /= len(mnist_test.dataset)
accuracy = 100.0 * correct / len(mnist_test.dataset)

# print the test result
print("\n")
print("end of training " + "\ttest loss: " + str(test_loss.item()) + " accuracy: " + str(accuracy.item()) + "%")

# the model sucked: 10% accuracy means that the model is no better than just randomly picking the label
# that 4th industrial revolution thingy is bs, time to learn blockchain

##########################################################################################
