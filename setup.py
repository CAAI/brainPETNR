#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='brainPETNR',
     version="0.1",
     author="Raphael Daveau & Claes Ladefoged",
     author_email="claes.noehr.ladefoged@regionh.dk",
     description="Brain PET noise reduction",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/CAAI/brainPETNR",
     scripts=[
             'inference_pipeline/brainPETNR',
     ],
     packages=find_packages(include=['inference_pipeline']),
     install_requires=[
         'rhscripts @ git+https://github.com/CAAI/rh-scripts.git@f316876afdc434107d8f442d93088aa07da6cff0#egg=rhscripts',
         'rhtorch @ git+https://github.com/CAAI/rh-torch.git@5007d29b04b0196031337ffc7c4c49aeed56b1cb#egg=rhtorch'
     ],
     classifiers=[
         'Programming Language :: Python :: 3.8',
     ],
 )