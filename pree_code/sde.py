# raw SDE for import
import torch
import torch.nn as nn
import torchsde

class NeuralSDE(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(NeuralSDE, self).__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.drift_nn = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.diffusion_nn = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def f(self, t, y):
        return self.drift_nn(y)
    def g(self, t, y):
        return self.diffusion_nn(y)
    def forward(self, ts, y0):
        return torchsde.sdeint(self, y0, ts, dt=0.01, method='euler')
