# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = ""
exec(open('text2vec2onnx/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='text2vec2onnx',
    version=__version__,
    description='Text to vector Tool, encode text',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='GanymedeNil',
    author_email='GanymedeNil@gmail.com',
    url='https://github.com/GanymedeNil/text2vec-onnx',
    license="Apache License 2.0",
    zip_safe=False,
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='word embedding,text2vec,onnx,Chinese Text Similarity Calculation Tool,similarity',
    install_requires=[
        "tokenizers",
        "numpy<=1.26.4",
    ],
    extras_require={
        "cpu": ["onnxruntime"],
        "gpu": ["onnxruntime-gpu"],
    },
    packages=find_packages(exclude=['tests']),
    package_dir={'text2vec2onnx': 'text2vec2onnx'},
    package_data={'text2vec2onnx': ['*.*', 'data/*.txt']}
)