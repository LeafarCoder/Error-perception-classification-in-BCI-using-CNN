# BCI-Feedback-Classification-using-CNN
Master thesis by Rafael Correia (2020)


# Index
- [Overview](#overview)
  - [Pre-processing](#pre-processing)
  - [Model training](#model-training)
- [Run experiments](#run-experiments)
  - [Setup](#setup)
- [Consult Results in Comet]
- [Reproduce results](#)
- [License](#license)

---
# Overview

## Pre-processing

## Model training



---
# Run experiments

The whole notebook document is organized in three main sections:
- [Setup](#setup)
- [Process dataset](#process-dataset)
- [Deep Learning Model](#deep-learning-model)


## Setup

In this section, the whole setup for the rest of the pre-processing and modeling is made.

First run *Necessary installs* to install all the required libraries to the Python environment in the remote Google Colab computer. Python will install the most recent updates on each package. If you notice one or more libraries are not interacting well with the rest then try downgrading the version by changing the code in *Necessary installs* to
```bash
pip install <library_name>==<version>
```

*Necessary installs* will kill the current runtime on purpose so the user does not forget to restart the runtime:
```bash
# Restart Google Colab session (ignore error)
import os
os.kill(os.getpid(), 9)
```

This is necessary to use the just installed libraries. Next, run *Necessary imports* and *Auxiliary functions* to add all necessary libraries and define the functions that will be used later on.

The dataset origin is defined in the next sub-section, *Define project directories*.
Here, two options are available, depending on how the user defines the variable on the first line, ```running_online```:

- Mount a Google Drive to access an online dataset (```running_online = True```)
  - Set the main folder (root): ```project_root_folder = "/content/drive/My Drive/Tese BCI/"```
  - After running this cell, access the given URL and choose your desired Drive.
  - Accept the terms and conditions, copy the authorization code to the prompt and press <Enter>
- Choose a local directory (```running_online = False```)
  - Organize the local directory to have the same structure as presented in the comments on the cell
  - Set the main folder (root): ```project_root_folder = "/any_local_directory/"```
  - Run the cell



## Process dataset

## Deep Learning Model

---
# License

Repository under MIT license.

You are free to use and modify the code present in this Github repository given that proper reference is made:
> R.   Correia,   “BCI   Feedback   Classification   using   CNN,”   2020.   [Online].   Available:https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN39



