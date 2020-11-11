import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='lola',
    version='0.0.1',
    packages=find_packages(),
    description='Learning with Opponent-Learning Awareness.',
    long_description=read('README.md'),
    license='MIT',
    install_requires=[
        'click', 'gym>=0.10.5', 'mock', 'numpy>=1.11', 'dm-sonnet==1.20', 'tensorflow>=1.8.0,<2.0.0',
        'trueskill', 'matplotlib', 'seaborn==0.9.0'
    ],
)
