# Reference: https://github.com/audreyr/cookiecutter-pypackage
# from distutils.core import setup
from setuptools import setup

setup(
    name='mpi4tf',
    packages=[],
    version='0.0.1',
    description='Tensorflow Based Distributed Training with MPI',
    author='Vibhatha Abeykoon',
    author_email='vibhatha@gmail.com',
    url='https://github.com/vibhatha/mpi4tf',
    keywords=['tensorflow', 'parallelism', 'mpi'],
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development',
    ],
    install_requires=[
        'tensorflow',
        'mpi4py',
        'numpy',
    ],
)
