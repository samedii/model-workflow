import torch

get_loss(dist, labels):
    return dist.log_prob(labels)
