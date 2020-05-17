# pytorch_template
A template for PyTorch projects with a CIFAR10 example.

To install and use:
0. (Create a Python3.7 environment, e.g. as conda create -n myenv python=3.7, and activate)
1. Adapt your project name and setup.py file
2. cd to the project root (where setup.py lives)
3. Install CUDA if necessary and install PyTorch through conda with the command specified by https://pytorch.org/. The tutorial was written using PyTorch 1.5.0. and CUDA10.2., so the command for Linux was 'conda install pytorch torchvision cudatoolkit=10.2 -c pytorch'
4. Execute 'pip install -r requirements.txt'
5. Execute 'pytest' to test the correct installation. Note that one of the tests tests whether at least one GPU is present. Some tests use the CIFAR10 data and are ignored if this data is not present.

To run the CIFAR10 experiment example:
1. Download CIFAR10 data in .png format from https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders
2. Execute sh scripts/cifar_10_experiment_example.sh
