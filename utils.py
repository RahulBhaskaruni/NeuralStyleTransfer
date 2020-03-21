from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def gram_matrix(mat):
    """
    :param mat: tensor for which gram matrix needs to be computed
    :return: gram matrix of the provided tensor
    """
    _, c, h, w = mat.size()
    mat = torch.Tensor.view(mat, (c, h*w))
    return torch.mm(mat, mat.t())


def get_content_activations(x, model, layer_name=['21']):
    """
    Input:
        x : the image we want to get the activation from
        layer_name : layer from which the activation needs to be extracted
                     defaulted to 21st layer assuming vgg19
        model : model from which activation needs to be extracted
    Output:
        activations for the given layer
    """
    x = x.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layer_name:
            return_this = x
    return return_this


def get_style_grams(x, model, layer_names):
    """
    Input:
        x : the image we want to get the activation from
        layer_names : layers from which the activation needs to be extracted
                     defaulted to 21st layer assuming vgg19
        model : model from which activation needs to be extracted
    Output:
        activation function for the given layer
    """
    x = x.unsqueeze(0)
    layer_grams = {}
    style_acts = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layer_names:
            style_acts[name] = x

    for name in layer_names:
        layer_grams[name] = gram_matrix(style_acts[name])
    return layer_grams


def get_loss(x, model, weights, style_grams, content_acts, content_layer_name=['21'], alpha=10, beta=40):
    """
    Function to calculate the total loss for neural style transfer
    :param x: current generated image
    :param model: model to be used for content and style calculation of activation
    :param weights: weights to be assigned when calculating style loss
    :param style_grams: gram matrices of the style layer chosen
    :param content_acts: content activations of the chosen layer
    :param content_layer_name: layer from which content is to be taken - same as one used in get_content_activations
    :param alpha:
    :param beta:
    :return: total loss including both style and content losses
    """
    loss = 0
    target_acts = {}
    x = x.unsqueeze(0)
    # getting necessary activations
    for name, layer in model._modules.items():
        x = layer(x)
        if name in list(weights.keys()):
            target_acts[name] = x
        if name in content_layer_name:
            target_content = x
    # style loss
    for name, wt in weights.items():
        style_gram = style_grams[name]
        _, c, h, w = target_acts[name].size()
        target_gram = gram_matrix(target_acts[name])
        loss += (weights[name] * torch.mean((target_gram - style_gram)**2)) / c * h * w
    # content loss
    style_loss = loss
    content_loss = torch.mean((content_acts - target_content) ** 2)
    return (alpha * content_loss) + (beta * style_loss)


def generate_noise(image):
    """
    Function to generate noise in the shape of the content image
    :param image: image in who's shape the noise needs to be generated
    :return: a tensor with noise
    """
    shape = image.size()
    noise = np.random.normal(0, 1 ** 0.5, shape).reshape(shape)
    return (torch.tensor(noise)).float()


def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)
    x = x * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    return x