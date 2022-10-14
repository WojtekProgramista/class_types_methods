from setuptools import setup, find_packages

setup(name='class_types_methods',
      version='1.0',
      description='Methods for analysing and transformnig imbalanced datasets',
      author='Wojciech Wieczorek',
      author_email='wojciech.wieczorek.new@gmail.com',
      url='https://github.com/WojtekProgramista/class_types_methods',
      packages=find_packages(),
      install_requires=["numpy>=1.13.3", "scipy>=0.19.1", "scikit-learn", "pandas"]
     )