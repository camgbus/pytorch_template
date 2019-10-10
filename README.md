# pytorch_template
A template for PyTorch projects with an Inception example.

To use:
1. Adapt your project name and setup.py file
2. cd to the project root (where setup.py lives)
3. Query your CUDA version with 'nvcc --version' (or install if necessary) and install PyTorch through conda with the command specified by https://pytorch.org/
4. Execute 'pip install -r requirements.txt'
5. Execute 'pytest' to test the correct installation

To tun the Inception example:
1. Download the HAM1000 dataset from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home
