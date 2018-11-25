import os
import argparse
from collections import OrderedDict


import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models


"""
使用 train.py 用数据集训练新的网络

基本用途：python train.py data_directory
在训练网络时，输出训练损失、验证损失和验证准确率
选项：
设置保存检查点的目录：python train.py data_dir --save_dir save_directory
选择架构：python train.py data_dir --arch "vgg13"
设置超参数：python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
使用 GPU 进行训练：python train.py data_dir --gpu
"""

def main():
    '''build our main function which calls different sub functions'''
    args = get_args()
    print("### Starting Main function ###")
    #load transformations
    dataloaders = transformations(args)

    #build model
    model = models.__dict__[args.arch](pretrained=True)
    model = build_classifier(model, args)

    #define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    # Print model and training args
    print_params(model, args)

    # train, test and save check point
    model = train(dataloaders, model, criterion, optimizer, args)
    model = test(dataloaders, model, criterion, args)
    checkpoint(model, dataloaders, optimizer, args)

def get_args():
    '''set up required arguments mentioned in review standard'''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('data_dir', metavar='data directory',help='directory to image datasets')
    parser.add_argument('--save_dir', help='Directory path to write checkpoint data', required=False)
    parser.add_argument('--arch', dest='arch',default='vgg13',help='Use vgg13 as default')
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='train model on gpu model')
    parser.add_argument('--epochs', dest='epochs', default=20,type=int, help='epoach numbers to run (default value =20)')
    parser.add_argument( '--learning_rate', dest='lr', default=0.01,type=float, help='learning rate (default value = 0.01)')
    parser.add_argument('--hidden_units', default =512, help='Hidden units argument for training (default value = 512)', required=False)
    

    return parser.parse_args()


def transformations(args):
    
    
	 #folder path
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    test_dir = os.path.join(args.data_dir, 'test')

    # Define transformations

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32,shuffle=True)
    test_dataloaders  = torch.utils.data.DataLoader(test_datasets, batch_size=32,shuffle=True)

    
    print('### Training Image are loaded ###')

    dataloaders = {}
    dataloaders['train'] = train_dataloaders
    dataloaders['test'] = test_dataloaders
    dataloaders['valid'] = valid_dataloaders
    dataloaders['train_datasets'] = train_datasets
    dataloaders['test_datasets'] = test_datasets
    dataloaders['valid_datasets'] = valid_datasets
    return dataloaders


def build_classifier(model, args):
 
   ## Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    if args.hidden_units:
        hidden_units = int(args.hidden_units)
    else:
        hidden_units = 512
        
    hidden_units_fc2 = int(hidden_units / 4)

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, hidden_units_fc2)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_units_fc2, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    return model


def print_params(model, args):
    '''for informational purpose'''
    print('### Args are below ###')
    print("   Using pre-trained model: ",args.arch)
    print('   Epochs: ', args.epochs)
    print('   Learning Rate: ', args.lr)
    print('   Hidden Units: ',args.hidden_units)


def train(dataloaders,model, criterion, optimizer, args):
    

    #GPU mode
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    print(" = training device mode: {} = ".format(device))  
    epochs = args.epochs
    steps = 0
    print_every = 30
    running_loss = 0

    for e in range(epochs):
        # Model in training mode, dropout is on
        print("Starting Training")
        model.train()
        for ii, (images, labels) in enumerate(dataloaders['train']):
            steps += 1
            # Flatten images into a 784 long vector
            #images.resize_(images.size()[0], 784)

            # Wrap images and labels in Variables so we can calculate gradients
            inputs = Variable(images)
            targets = Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                test_loss = 0
                with torch.no_grad():
                    for ii, (images, labels) in enumerate(dataloaders['test']):
                        #images = images.resize_(images.size()[0], 784)
                        # Set volatile to True so we don't save the history
                        inputs = Variable(images)
                        labels = Variable(labels)
                        inputs, labels = inputs.to(device), labels.to(device)

                        output = model.forward(inputs)
                        test_loss += criterion(output, labels).data[0]

                        ## Calculating the accuracy 
                        # Model's output is log-softmax, take exponential to get the probabilities
                        ps = torch.exp(output).data
                        # Class with highest probability is our predicted class, compare with true label
                        equality = (labels.data == ps.max(1)[1])
                        # Accuracy is number of correct predictions divided by all predictions, just take the mean
                        accuracy += equality.type_as(torch.FloatTensor()).mean()




                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['test'])),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))

                running_loss = 0

                # Make sure dropout is on for training
                model.train()
    print("Training Completed!")

    return model


def test(dataloaders, model, criterion, args):
    
    model.to('cuda') 
    model.eval()
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        for ii, (images, labels) in enumerate(dataloaders['test']):

            inputs = Variable(images)
            labels = Variable(labels)
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            output = model.forward(inputs)
            test_loss += criterion(output, labels).data[0]

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output).data
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    print( "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['test'])),
                "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
    
    return model

def checkpoint(model, dataloaders, optimizer, args):
    ''''''
    # Switch to CPU for loading compatability
    model.to('cpu')

    # Map classes to model indices
    model.class_to_idx = dataloaders['train_datasets'].class_to_idx

    checkpoint = {
        'arch' :args.arch,
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'hidden_units' : args.hidden_units
    }   
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = '.' #current directory
 
    checkpoint_path = save_dir + '/checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint is saved at:", checkpoint_path)


if __name__ == '__main__':
    main()