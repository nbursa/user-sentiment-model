from setuptools import setup, find_packages

setup(
    name='user-sentiment-model',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'tensorflow',
    ],
    entry_points={
        'console_scripts': [
            'main=src.main:main',
        ],
    },
)
