import mnist_problem
from .model import Model
from .config import Config


if __name__ == '__name__':
    mnist_problem.web_api(Config, Model)
