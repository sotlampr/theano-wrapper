from setuptools import setup

setup(name='theano_wrapper',
      version='0.0a',
      description='Utilities for machine learning applications with theano',
      url='https://github.com/sotlampr/theano-wrapper',
      author='Sotiris Lamprinidis',
      author_email='sot.lampr@gmail.com',
      license='MIT',
      packages=['theano_wrapper'],
      install_requires=['numpy', 'scipy', 'theano', 'scikit-learn',
                        'nose', 'coverage'],
      zip_safe=False)


