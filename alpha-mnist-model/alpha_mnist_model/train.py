import didactic_meme as model_suite
from .config import Config
import alpha_mnist_model

# python -m alpha_mnist_model.train path/to/model_dir
train = model_suite.command_line.train(Config, alpha_mnist_model.model.train)
