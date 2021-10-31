# ML-Project1

> This repository contains the code for the first project for the Machine
> Learning class (CS-433) at EPFL.

The goal of the project is to implement from scratch the ML concepts seen in 
class on a real-world dataset: data from CERN about the Higgs boson.

## Run code

To reproduce our results, you will need to run the scripts `scripts/run.py`. For this, you will need to follow the steps we describe in this section. Run all commands from the root of the repository.

### Unzip datasets

First you need to unzip the training and testing datasets.

```bash
cd data/
unzip test.csv.zip
unzip train.csv.zip
cd ..
```

### Install the requirements

To make the script work, you will need to install the package `numpy`. You can either use `pip` or `conda` to install it.

```
conda create --name ml-project1
conda activate ml-project1
conda install numpy -y
```

### Run the script

Finally, you can run the script.

```
python3.9 scripts/run.py 
```

This creates the file `prediction.csv` which contains the results of our model on the test dataset.
