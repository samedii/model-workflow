import didactic_meme as model_suite
from .config import Config


# python -m alpha_mnist_model.create path/to/model_dir
if __name__ == '__main__':
    model_suite.command_line.create(Config)
