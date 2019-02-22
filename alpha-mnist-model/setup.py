from setuptools import setup

setup(
   name='alpha-mnist-model',
   version='0.1',
   description='mnist model example for model workflow',
   author='Richard Löwenström',
   author_email='samedii@github.com',
   packages=['alpha_mnist_model'],
   install_requires=['mnist-problem', 'torch', 'torchvision', 'didactic-meme'],
)
