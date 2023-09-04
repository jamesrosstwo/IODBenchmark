conda create --name IODBench python=3.7
conda activate IODBench

conda install -c menpo opencv
conda install -c anaconda matplotlib numpy pandas scikit-learn pyyaml scipy cython
conda install -c conda-forge tabulate termcolor tqdm wandb pyhocon pycocotools tensorboard munch
conda install -c conda-forge gxx_linux-64==5.4.0

