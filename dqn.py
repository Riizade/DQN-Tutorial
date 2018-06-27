import gym
import math
import random
import numpy
import matplotlib
import matplotlib.pyplot
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn
import torch.optim
import torch.nn.functional
import torchvision.transforms

# set up gym environment
env = gym.make("CartPole-v0").unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
matplotlib.pyplot.ion()

# set up whether we're using CUDA/GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")