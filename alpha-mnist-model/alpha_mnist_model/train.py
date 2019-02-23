import os
import torch
import torch.utils.data
import tensorboardX as tbx
import logging
logger = logging.getLogger(__name__)


import didactic_meme as model_suite
import mnist_problem
from .config import Config
from .model import Model


def train(config):
    model_suite.setup_loggers(config)

    torch.manual_seed(config.seed)

    device = torch.device('cuda' if config.cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}
    train_loader = torch.utils.data.DataLoader(mnist_problem.get_dataset('train'),
        batch_size=config.batch_size, shuffle=True, **kwargs)
    validate_loader = torch.utils.data.DataLoader(mnist_problem.get_dataset('validate'),
        batch_size=config.eval_batch_size, shuffle=True, **kwargs)

    model = Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    tb = tbx.SummaryWriter(log_dir=config.model_dir) # TODO: examples
    save_dir = os.path.join(config.model_dir, 'save') # TODO: move to model-suite?
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, config.n_epochs + 1):
        train_epoch(config, model, device, train_loader, optimizer, epoch)
        eval_epoch(config, model, device, validate_loader)

        n_epochs_chars = len(str(config.n_epochs))
        torch.save(dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        ), os.path.join(save_dir, f'epoch{epoch:0{n_epochs_chars}d}.pth'))


def train_epoch(config, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        dist = model(data)
        # loss = mnist_problem.metrics.get_loss(dist, target).mean()
        loss = -dist.log_prob(target).mean()
        loss.backward()
        optimizer.step()
        if batch_idx % config.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def eval_epoch(config, model, device, eval_loader):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            dist = model(data)
            # eval_loss += mnist_problem.metrics.get_loss(dist, target).sum().item()
            eval_loss += dist.log_prob(target).sum()
            # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss /= len(eval_loader.dataset)

    # logger.info('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     eval_loss, correct, len(eval_loader.dataset),
    #     100. * correct / len(eval_loader.dataset)))


# python -m alpha_mnist_model.train path/to/model_dir
if __name__ == '__main__':
    model_suite.command_line.train(Config, train)
