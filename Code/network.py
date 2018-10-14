import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import utils
from torchvision import transforms
import numpy as np
import pandas as pd
from timeit import timeit
import matplotlib
matplotlib.use('Agg')  # use agg backend for server
import matplotlib.pyplot as plt

# set hyper parameters
batch_size = 100
input_chan = 4
kernel_size = 4
learning_rate = 5e-3
momentum = 0.9
#weight_decay = 5e-4
epochs = 10
input_size = 28

##########################
# CLASSES
##########################

# define dataset class
class DeepSAT6(Dataset):
    '''DeepSat6 Dataset'''

    def __init__(self, img_file, label_file, transform=None):
        self.img = pd.read_csv(img_file, header=None)
        self.label = pd.read_csv(label_file, header=None)
        self.transform = transform

    # built-in length
    def __len__(self):
        return len(self.img)

    # fetch a row from the csv files for the image and label
    # reshape the image row to numpy image format (image has 4 channels: RGB and near-infrared)
    # if a transform is passed, apply it and return the sample
    def __getitem__(self, index):
        img = self.img.iloc[index,:]
        img = img.values.reshape((input_size, input_size, 4))
        label = self.label.iloc[index,:]
        # convert labels from one-hot encoding to class number, by returning index of 1-value
        label = label.idxmax()
        label = int(label)
        sample = img, label

        if self.transform:
            sample = self.transform(sample)

        return sample

# define tensor transformation class
class ToTensor(object):
    '''Converts data in numpy arrays to tensor and reshapes images to CxHxW order'''

    def __call__(self, sample):
        img, label = sample

        # reverse order of axes from numpy image to tensor image format
        # normalize into the range [0,1]
        img = img.transpose((2,0,1))
        #img = img.astype(float)
        img = (img + 1)/256
        # convert image to Tensor, leave label as int.
        # DataLoader will convert label to Tensor of size=batch_size
        return torch.FloatTensor(img), label

# convolution output size is given by W x H X C,
# where W = H = [(input size - kernel size + 2(padding))/stride] + 1 [assuming square input]
# C is number of channels or input feature maps
class SimpleCNN(nn.Module):
    '''Neural network'''

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_chan, out_channels=16, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=12 * 12 * 16, out_features=num_classes)

    def forward(self, p):
        self.input = p
        self.features = self.conv1(p)
        a = f.relu(self.features)
        a = self.pool(a)
        a = a.view(-1, 12 * 12 * 16)
        a = self.fc1(a)
        return a

class QuasiAlex(nn.Module):
    '''A simplified version of AlexNet'''

    def __init__(self):
        super(QuasiAlex, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_chan, out_channels=12, kernel_size=11)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=6 * 6 * 32, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=24)
        self.fc3 = nn.Linear(in_features=24, out_features=6)

    def forward(self, p):
        a = f.relu(self.conv1(p))
        a = f.relu(self.conv2(a))
        a = f.relu(self.conv3(a))
        a = f.relu(self.conv4(a))
        a = f.relu(self.conv5(a))
        a = self.pool(a)
        a = a.view(-1, 6 * 6 * 32)
        a = f.dropout(f.relu(self.fc1(a)))
        a = f.dropout(f.relu(self.fc2(a)))
        a = self.fc3(a)
        return a

##########################
# DATA SETUP
##########################
print('Loading datasets...')
# instantiate training and testing datasets
train_set = DeepSAT6(img_file='../Dataset/X_train_sat6.csv',
    label_file='../Dataset/y_train_sat6.csv',
    transform=transforms.Compose([ToTensor()]))

test_set = DeepSAT6(img_file='../Dataset/X_test_sat6.csv',
    label_file='../Dataset/y_test_sat6.csv',
    transform=transforms.Compose([ToTensor()]))

# load datasets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# label class names
classes = ['building', 'barren land', 'trees', 'grassland', 'road', 'water']
num_classes = len(classes)

# save sample image to file and print label to console
train_iter = iter(train_loader)
image, label = train_iter.next()
# save sample image grids as two files (RBG & near infrared)
utils.save_image(image[:,:3,:,:], '../Plots/sampleRGB.png')
utils.save_image(image[:,[3],:,:],'../Plots/sampleNIR.png') # preserve Bat x C x H x W

# print class names of images in grid (# of images = batch_size)
print('Sample image grid saved to /Plots with classes:')
print(', '.join(classes[label[i]] for i in range(batch_size)))

##########################
# NETWORK SET-UP
##########################

# instantiate Network
network = SimpleCNN()
#network = QuasiAlex()
# transfer to GPU
network.cuda()

# print network details to console
print('NETWORK PARAMETERS:\n\
Batch size: {:0.0f}\n\
Epochs: {:0.0f}\n\
Learning rate: {:f}\n\
Momentum: {:f}\n\
Input channels: {:0.0f}\n\
Input dimensions: {:0.0f}\n\
Conv kernel size: {:0.0f}'
    .format(batch_size, epochs, learning_rate, momentum,input_chan,
    input_size, kernel_size))

print('NETWORK CONFIGURATION:')
for i, m in enumerate(network.modules()):
    print(i, '->', m)

# define loss and optimiser
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

##########################
# TRAIN NETWORK
##########################

# track the training loss at each iteration and training & testing accuracy at each epoch
running_loss = np.zeros((epochs, len(train_loader)))
running_train_acc = np.zeros((epochs))
running_test_acc = np.zeros((epochs))
# set-up confusion matrix of training accuracy
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

def train():

    for epoch in range(epochs):

        # rest running totals of correct guesses
        train_correct = 0
        test_correct = 0

        for i, data in enumerate(train_loader):

            # get data and convert to Variable
            images, labels = data

            # zero gradient buffer
            optimizer.zero_grad()

            # forward pass
            output = network(Variable(images.cuda()))
            # calculate loss
            error = loss(output, Variable(labels.cuda()))
            # backprop
            error.backward()
            # optimization step
            optimizer.step()

            # get network prediction (neuron index with largest output)
            train_predict = torch.max(output.data, 1)[1]

            # compare each prediction in batch to label and increment test_correct tally
            train_correct += (train_predict.cpu() == labels).sum()

            # accuracy at each epoch: add to tracking
            running_train_acc[epoch] = train_correct / len(train_set) * 100

            # show training loss at every 100 data points
            if (i + 1) % 100 == 0:
                print('Training loss at epoch {} of {}, step {} of {}: {:.4f}'.format(
                epoch + 1, epochs, (i + 1), len(train_loader), error.data[0]
                ))

            # add to running loss
            running_loss[epoch, i] = error.data[0]


        # run testing set through network at each epoch
        for data in test_loader:

            # get data and convert images to Variable
            images, labels = data

            # calculate output
            output = network(Variable(images.cuda()))

            # get network prediction (neuron index with largest output)
            test_predict = torch.max(output.data, 1)[1]

            # compare each prediction in batch to label and increment test_correct tally
            test_correct += (test_predict.cpu() == labels).sum()

            # last epoch only
            # for each item in the batch, increment by one the corresponding row and column
            if epoch == epochs - 1:
                for i in range(batch_size):
                    confusion_matrix[test_predict[i], labels[i]] += 1

        # accuracy at each epoch: add to tracking, print
        running_test_acc[epoch] = test_correct / len(test_set) * 100

        print('Accuracy of network on test set at epoch {} of {}: {}/{} = {:.2f}%'.format(
        epoch + 1, epochs, test_correct, len(test_set), test_correct / len(test_set) * 100
        ))

##########################
# RUN TRAINING AND TRACK TIME
##########################

print('Training time: {:0.1f} seconds.'.format(timeit(train, number=1)))

##########################
# SHOW OTHER NETWORK METRICS
##########################

running_loss = running_loss.reshape(-1)

# plot running loss
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(running_loss, alpha=0.7)
# label x axis with epochs using formatter
ax.set_xticklabels(['{:0.0f}'.format(i) for i in ax.get_xticks() / (len(train_set)/batch_size)])
ax.set_title('Trianing Loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
fig.savefig('../Plots/training_loss.png')
print('Training loss plot saved in /Plots.')

# plot comparison of testing and training accuracy
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(running_test_acc, label='Testing Accuracy', alpha=0.8)
ax.plot(running_train_acc, label='Training Accuracy', alpha=0.8)
#ax.set_ylim(0, 100)
ax.set_xticklabels('{:0.0f}'.format(i+1) for i in ax.get_xticks())
ax.set_title('Predication Accuracy')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Epoch')
ax.legend(loc='lower right', fontsize='medium')
fig.savefig('../Plots/testing_accuracy.png')
print('Testing Accuracy plot saved in /Plots.')

# confusion matrix of predictions
# convert to dataframe and add column, index names
confusion_matrix_df = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
print(confusion_matrix_df)

# calculate accuracy and misclassification rate
accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum() * 100
wrong = 100 - accuracy
print('Accuracy rate: {:.2f}%\nMisclassifcation rate: {:.2f}%'.format(accuracy, wrong))

# plot confusion confusion
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[8,8])
ax.matshow(confusion_matrix, cmap='summer', vmin=confusion_matrix.min(),
            vmax=confusion_matrix.max())
ax.set_xlim(-1, 6)
ax.set_ylim(6, -1) # reverse order
ax.set_xticklabels(classes, rotation=90, size='x-small')
ax.set_xticks([i for i in range(len(classes))])
ax.set_yticklabels(classes, size='x-small')
ax.set_yticks([i for i in range(len(classes))])
ax.set_xlabel('Actual', size='small')
ax.set_ylabel('Predicted', size='small')

# label cells
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(i, j, confusion_matrix[j, i], ha='center', va='center', color='black')

fig.savefig('../Plots/confusion_matrix.png')
print('Confusion matrix plot saved in /Plots.')

#Plot all kernels


def plot_kernels(kernels, num_cols, file_name, title):
    num_kernels = kernels.shape[0]
    num_rows = 1 + num_kernels
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(num_kernels):
        ax = fig.add_subplot(num_rows,num_cols, i+1)
        ax.imshow(kernels[i].transpose(),interpolation='nearest')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.suptitle(title)
    fig.savefig('../Plots/' + file_name + '.png')
    print('Kernels plot saved in /Plots.')


plot_kernels(network.conv1.weight.data.cpu().numpy(), 4, 'conv1_kernels', 'Convolutional Layer 1 Kernels')

#Plot sample feature map


def plot_feature_map(feature_maps, num_cols, file_name, title):
    num_maps = feature_maps.shape[0]
    num_rows = 1 + num_maps
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_maps):
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.imshow(feature_maps[i])
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('Filter ' + str(i+1), fontsize=8)
    fig.suptitle(title)
    fig.savefig('../Plots/' + file_name + '.png')
    print('Feature maps saved in /Plots.')

plot_feature_map(network.features.data.cpu().numpy()[0], 4, 'feature_maps', 'Feature Maps')

