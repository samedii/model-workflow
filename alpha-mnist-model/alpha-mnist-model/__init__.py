import model_suite
from .config import Config
import alpha_mnist_model.model

# python -m alpha_mnist_model.create path/to/model_dir
create = model_suite.create(Config)

# python -m alpha_mnist_model.train path/to/model_dir
train = model_suite.train(Config, alpha_mnist_model.model.train)
