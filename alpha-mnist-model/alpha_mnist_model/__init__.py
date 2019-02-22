import didactic_meme as model_suite
from .config import Config
import alpha_mnist_model.model

# python -m alpha_mnist_model.create path/to/model_dir
# create = model_suite.command_line.make_create(Config)

# python -m alpha_mnist_model.train path/to/model_dir
# train = model_suite.command_line.make_train(Config, alpha_mnist_model.model.train)
