from setuptools import setup, find_packages

setup(
    name='tensortrace',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.13.0',
        'numpy>=1.20',
        'h5py>=3.0.0',
        'typing-extensions; python_version < "3.8"',
    ],
    python_requires='>=3.7',
    author='AndyBarcia',
    author_email='andybarcia4@gmail.com',
    description='A PyTorch utility for tracing and saving model variables during execution.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AndyBarcia/tensortrace',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='pytorch, tracing, model analysis, debugging, hdf5, tensor',
)

    