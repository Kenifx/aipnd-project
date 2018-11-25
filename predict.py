import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict

'''使用 predict.py 预测图像的花卉名称以及该名称的概率

基本用途：python predict.py input checkpoint
选项：
返回前 KK 个类别：python predict.py input checkpoint --top_k 3
使用类别到真实名称的映射：python predict.py input checkpoint --category_names cat_to_name.json
使用 GPU 进行训练：python predict.py input checkpoint --gpu
'''


################################################################################################################


def main():
    '''build our main function which calls different sub functions'''
    args = get_args()
    print("### Starting Main function ###")
    
    checkpoint = args.checkpoint
    input_path = args.input
    
    #load check point
    print('load checkpoint')   
    model = load(checkpoint,args)
    
    #process image
    process_image(input_path)
    
    #predict
    predict(input_path, model, topk=3)
    
    #mapping
    if args.top_k:
        topk = ints(args.top_k)
    else:
        topk = 3
        
    probs, classes = predict(input_path, model, topk=3)  
    map_label(args, probs, classes)


def get_args():
    '''set up required arguments mentioned in review standard'''
    parser = argparse.ArgumentParser(description='')

    parser = argparse.ArgumentParser(description='predict.py: Opens a pretrained network and predicts an image class.')
    parser.add_argument('--input', help='Input image file for prediction', required=True)
    parser.add_argument('--checkpoint', help='File path to checkpoint', required=True)
    parser.add_argument('--top_k', help='Display Topk', required=False)
    parser.add_argument('--category_names', help='Mapping category label to actual label', required=False)
    parser.add_argument('--gpu', help='use GPU for training', required=False)

    return parser.parse_args()


def load(checkpoint, args):
    checkpoint = torch.load(checkpoint) 
    
    #GPU or CPU mode + check gpu availability
    if args.gpu and  torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
          
    #training model
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    hidden_units = checkpoint['hidden_units']
    hidden_units_fc2 = int(hidden_units / 4)
    
    #make input size as a variable instead of hard coded value
    input_size = model.classifier[0].in_features
    
    print("Building Classifer ")
    
    
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, hidden_units_fc2)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_units_fc2, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]  
    
    pil_image = Image.open(image).convert("RGB")
    
    data_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    pil_image = data_transforms(pil_image)
    return pil_image


def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    model.to('cpu') #Else will get error Expected object of type torch.cuda.FloatTensor but found type torch.FloatTensor for argument #4 'mat1'
    # load image 
    image = process_image(image_path)
    image = image.unsqueeze(0) #for VGG model
    
    with torch.no_grad():
        output = model.forward(image)
        probabilities, labels = torch.topk(output, topk)
        
        probabilities = probabilities.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = list()
    
    for i in labels.numpy()[0]:
        classes.append(class_to_idx_inv[i])
        
    return probabilities.numpy()[0], classes



################################################################################################################
###### Map the labels

def map_label(args, probs, classes):
    if args.category_names:
        category_names = args.category_names
    else:
        category_names = 'cat_to_name.json'
           
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    labels = []
    '''for i in classes:
        s = str(i + 1)
        labels.append(cat_to_name[s])'''
        
    for class_idx in classes:
        labels.append(cat_to_name[class_idx])

    df = pd.DataFrame({'Classes':classes, 'Prob':probs, 'Labels':labels})

    top = df.sort_values(by=['Prob'], ascending=0)

    print("Top " + str(len(classes)) + " possible labels for the image:")
    print(str(top))

if __name__ == '__main__':
    main()
