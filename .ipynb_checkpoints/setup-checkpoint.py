from setuptools import setup

setup(
    name='stochastic_control',
    version='0.1',
    license='MIT',
    description='Python code for Reinforcement Learning and Control',
    long_description=open('README.md').read(),
    author='Neil Walton',
    author_email='neil.walton@appliedprobability.blog',
    classifiers=['Reinforcement Learning','Stochastic Control'],
    install_requires=['numpy']
)