# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.10, CUDA 12.1

conda deactivate

# Set environment variables
export ENV_NAME=void_sfm
export PYTHON_VERSION=3.10
export PYTORCH_VERSION=2.1.0
export CUDA_VERSION=12.1

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install PyTorch, torchvision, and PyTorch3D using conda
conda install pytorch=$PYTORCH_VERSION torchvision pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath # Ori
# https://anaconda.org/conda-forge/fvcore
#conda install conda-forge::fvcore
# https://anaconda.org/conda-forge/iopath
#conda install conda-forge::iopath
# https://anaconda.org/pytorch3d/pytorch3d
# https://stackoverflow.com/questions/77401881/cannot-install-pytorch3d-using-conda-or-pip-on-windows-11
# pytorch3d need to be installed from source
#pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
#conda install pytorch3d::pytorch3d
conda install pytorch3d -c pytorch3d # Ori

# Install pip packages
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom tqdm
pip install accelerate==0.24.0

# Install LightGlue
git clone https://github.com/jytime/LightGlue.git dependency/LightGlue

cd dependency/LightGlue/
python -m pip install -e .  # editable mode
cd ../../

# Force numpy <2
pip install numpy==1.26.3

# Ensure the version of pycolmap is 0.6.1
pip install pycolmap==0.6.1

# (Optional) Install poselib 
pip install poselib==2.0.2

