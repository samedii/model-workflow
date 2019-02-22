import torch

def get_loss(dist, labels):
    return dist.log_prob(labels)
