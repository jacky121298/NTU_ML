import os
import sys
import copy
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import models

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is (3 x W x H)
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape (1 x W x H) or (W x H) or (3 x W x H)
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis = 0)
    
    # Phase/Case 2: The np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert (1 x W x H) to (3 x W x H)
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis = 0)
    
    # Phase/Case 3: The np arr is of shape (3 x W x H)
    # Result: Convert it to (W x H x 3) in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    
    # Phase/Case 4: The np arr is normalized between 0 ~ 1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape (C x W x H)
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def preprocess_image(pil_im, resize_im = True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((128, 128), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to (C, W, H)
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = (1, 3, 224, 224)
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad = True)
    return im_as_var

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im)
    return recreated_im

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def building_block(indim, outdim):
            return [
                nn.Conv2d(indim, outdim, 3, 1, 1),
                nn.BatchNorm2d(outdim),
                nn.PReLU(),
            ]

        def stack_blocks(indim, outdim, block_num):
            layers = building_block(indim, outdim)
            for i in range(block_num - 1):
                layers += building_block(outdim, outdim)
            layers.append(nn.MaxPool2d(2, 2, 0))
            return layers

        cnn_list = []
        cnn_list += stack_blocks(3, 64, 2)
        cnn_list += stack_blocks(64, 128, 1)
        cnn_list += stack_blocks(128, 256, 1)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential(*cnn_list)

        dnn_list = [
            nn.Linear(512 * 4 * 4, 1024),
            nn.Dropout(p = 0.5, inplace = True),
            nn.PReLU(),

            nn.Linear(1024, 512),
            nn.Dropout(p = 0.5, inplace = True),
            nn.PReLU(),
            nn.Linear(512, 11)
        ]
        self.fc = nn.Sequential(*dnn_list)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class DeepDream():
    """
        Produces an image that minimize the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, im_path):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.created_image = Image.open(im_path).convert('RGB')
        self.hook_handle = self.hook_layer()

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        hook_handle = self.model.fc[self.selected_layer].register_forward_hook(hook_function)
        return hook_handle

    def dream(self, output_img, iteration):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, True)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr = 0.04,  weight_decay = 1e-4)
        
        for iter in range(iteration):
            optimizer.zero_grad()
            # Assign image to a variable to move forward in the model
            x = self.processed_image.cuda()
            for layer in self.model.cnn:
                x = layer(x)

            x = x.view(x.size()[0], -1)
            for index, layer in enumerate(self.model.fc):
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    break
            
            # Loss function is the mean of the output of the selected layer/filter
            # We try to maximize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            loss.backward()
            optimizer.step()
            print('[ Iteration: ' + str(iter + 1) + ' ] Loss:', '{0:.2f}'.format(loss.data.cpu().numpy()), end = '\r')

            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image every 100 iteration
            if (iter + 1) % 100 == 0:
                print('\nsave image iter = {}'.format(iter + 1))
                im_path = '{}/dream'.format(output_img) + '_iter' + str(iter + 1) + '.jpg'
                save_image(self.created_image, im_path)

if __name__ == '__main__':
    im_path = sys.argv[1]
    model_path = sys.argv[2]
    output_img = sys.argv[3]

    dnn_layer = 6
    filter_pos = 6
    iteration = 1000

    print("Loading model ...")
    model = Classifier().cuda()
    model.load_state_dict(torch.load(model_path))

    dd = DeepDream(model, dnn_layer, filter_pos, im_path + '/training/0_8.jpg')
    dd.dream(output_img, iteration)
    dd.hook_handle.remove()