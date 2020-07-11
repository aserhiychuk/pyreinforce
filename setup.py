from setuptools import setup, find_packages

setup(
    name='PyReinforce',
    version='0.2',
    description='Deep Reinforcement Learning library for Python',
    author='Anton Serhiychuk',
    url='https://github.com/aserhiychuk/pyreinforce',
    license='MIT',
    install_requires=[
        'numpy>=1.0'
    ],
    extras_require={
        'gym': ['gym'],
        'tensorflow': ['tensorflow>=1.0']
    },
    packages=find_packages()
)
