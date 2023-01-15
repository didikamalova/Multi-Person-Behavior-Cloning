import torch
import numpy as np
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy, MlpPolicy
from rlkit.torch.networks import CNN, TwoHeadDCNN, DCNN
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(

            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens)
             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs, ):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class ImagePolicy(nn.Module):
    def __init__(
            self,

            x_dim=48,
            y_dim=48,
            input_channels=3,
            action_size=4,
            
            embedding_dim=3,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64):
            
        super(ImagePolicy, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.x_dim, self.y_dim = x_dim, y_dim
        self.imsize = x_dim * y_dim
        self.imlength = self.imsize * self.imsize * input_channels

        self._encoder = Encoder(input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        self._final_conv = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)

        # Calculate latent sizes
        if x_dim == 96 and y_dim == 128:
            self.discrete_x, self.discrete_y = 24, 32
        if x_dim == 48 and y_dim == 48:
            self.discrete_x, self.discrete_y = 12, 12

        self.discrete_size = self.discrete_x * self.discrete_y
        self.representation_size = self.discrete_size * self.embedding_dim
        
        self.num_layers = 4
        self.num_hiddens = 400
        self.num_people = 5
        self.hidden_layers = nn.ModuleList([])
        
        # Define BC network
        # Defining the input and output dimensions 
        for i in range(self.num_layers):
            if i == 0:
                in_dim, out_dim = self.representation_size + self.num_people, self.num_hiddens
            elif i == (self.num_layers-1):
                in_dim, out_dim = self.num_hiddens, action_size
            else:
                in_dim, out_dim = self.num_hiddens,  self.num_hiddens

            curr_layer = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(curr_layer.weight, gain=1)
            curr_layer.bias.data.uniform_(-1e-3, 1e-3)
            
            self.hidden_layers.append(curr_layer)

        self.relu = nn.LeakyReLU()

    def compute_loss(self, obs, expert_actions, person_id):
        combined_latent = self.encode(obs, person_id)
        action = self.get_action(combined_latent)
        bc_loss = F.mse_loss(action, expert_actions)
        return bc_loss

    def process_image(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.x_dim, self.y_dim)

        z = self._encoder(inputs)
        z = self._final_conv(z)

        return z.reshape(-1, self.representation_size)

    def encode(self, obs, person_id):
        latent = self.process_image(obs)
        person_id = person_id.reshape(-1, self.num_people)
        combined_latent = torch.cat([latent, person_id], 1)
        return combined_latent

    #policy input, output of encode
    def get_action(self, combined_latent):
        for i in range(self.num_layers):
            combined_latent = self.hidden_layers[i](self.relu(combined_latent))
        return torch.tanh(combined_latent)

    def forward(self, obs, person_id):
        combined_latent = self.encode(obs, person_id)
        action = self.get_action(combined_latent)
        return action
