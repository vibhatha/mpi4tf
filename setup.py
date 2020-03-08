# Reference: https://github.com/audreyr/cookiecutter-pypackage
#from distutils.core import setup
from setuptools import setup
setup(
    name='tfpipegrad',
    packages=[],
    version='0.0.1',
    description='Tensorflow Based Pipeline Training with Autograd Optimization',
    author='Vibhatha Abeykoon',
    license='Apache License 2.0',
    author_email='vibhatha@gmail.com',
    url='https://github.com/vibhatha/tfpipegrad',
    keywords=['tensorflow', 'pipeline', 'parallelism', 'autograd' ],
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: Apache License 2.0',
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
)