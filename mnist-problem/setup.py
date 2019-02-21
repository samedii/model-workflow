from setuptools import setup

setup(
   name='mnist-problem',
   version='0.1',
   description='mnist problem example for model workflow',
   author='Richard Löwenström',
   author_email='samedii@github.com',
   packages=['mnist_problem'],
   install_requires=['torch', 'torchvision'],
)
