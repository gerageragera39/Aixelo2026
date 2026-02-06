import random
import numpy as np
import torch
import torch.optim as optim


# Constants
INITIAL_LEARNING_RATE = 0.01
LR_DECAY_FACTOR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_optimizer(model_parameters, lr=0.1):
    return optim.SGD(model_parameters, lr=lr,
                     momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY)

