import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img2tensor(img,image_size = 128):

    image = img
    if max(image.size) <=image_size:
        image_size = max(image.size)

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                               torchvision.transforms.ToTensor()])
    img = transform(image).unsqueeze(0)

    return img.to(device, torch.float)


def gram_matrix(inp):

    """
    This function takes in the input of the batch 

    Output : Normalized Gram Matrix
    """

    num_batch, num_channels, height, width = inp.size()
    feats = inp.view(num_batch * num_channels, width * height)
    gram_mat = torch.mm(feats, feats.t()) 
    return gram_mat.div(num_batch * num_channels * width * height)
    

def model_mods():
    vgg19 = torchvision.models.vgg19(weights="IMAGENET1K_V1").to(device)
    vgg19_model = vgg19.features
    for param in vgg19_model.parameters():
        param.requires_grad_(False)
    
    conv_indices = []

    for i in range(len(vgg19_model)):
        if vgg19_model[i]._get_name() == 'MaxPool2d':
            vgg19_model[i] = nn.AvgPool2d(kernel_size=vgg19_model[i].kernel_size, 
                                      stride=vgg19_model[i].stride, 
                                      padding=vgg19_model[i].padding)
        if vgg19_model[i]._get_name() == 'Conv2d':
            conv_indices.append(i)
        
    conv_indices = dict(enumerate(conv_indices, 1))

    return vgg19_model, conv_indices

def model_train(img, styl, wt_style, wt_content, epochs):

    img = img2tensor(img)
    styl = img2tensor(styl)

    layers = {1: 's', 2: 's', 3: 's', 4: 'sc', 5: 's'}
    vgg19_model, conv_indices = model_mods()
    vgg_layers = nn.ModuleList(vgg19_model)
    last_layer_idx = conv_indices[max(layers.keys())]
    vgg_layers_trimmed = vgg_layers[:last_layer_idx+1]
    neural_style_transfer_model = nn.Sequential(*vgg_layers_trimmed)
    inp_image = torch.randn(img.size()).to(device)
    opt = optim.Adam([inp_image.requires_grad_()], lr=0.01)

    for curr_epoch in range(1, epochs+1):
        inp_image.data.clamp_(0, 1)
        opt.zero_grad()
        epoch_style_loss = 0
        epoch_content_loss = 0

        for k in layers.keys():
            if 'c' in layers[k]:
                target = neural_style_transfer_model[:conv_indices[k]+1](img).detach()
                ip = neural_style_transfer_model[:conv_indices[k]+1](inp_image)
                epoch_content_loss += torch.nn.functional.mse_loss(ip, target)
            if 's' in layers[k]:
                target = gram_matrix(neural_style_transfer_model[:conv_indices[k]+1](styl)).detach()
                ip = gram_matrix(neural_style_transfer_model[:conv_indices[k]+1](inp_image))
                epoch_style_loss += torch.nn.functional.mse_loss(ip, target)

        epoch_style_loss *= wt_style
        epoch_content_loss *= wt_content
        total_loss = epoch_style_loss + epoch_content_loss
        total_loss.backward()
        opt.step()

    return inp_image.data.clamp_(0, 1).squeeze(0).cpu().detach().numpy().transpose(1,2,0)

