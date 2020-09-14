# BCI-Feedback-Classification-using-CNN
Master thesis by Rafael Correia (2020)


# Index
- [Overview](#overview)

- [Run experiments](#run-experiments)
  - [Setup](#setup)
  - [Process dataset](#process-dataset)
  - [Deep Learning Model](#deep-learning-model)
- [Consult Results in Comet](#)
- [License](#license)
- [BibTeX](#bibtex)

---
# Overview

...



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

There are several processing steps that can be aplied to the dataset:
- Raw data conversion
  - Raw data is converted from Comma Separated Value files (.csv) files into Pickle files (.p)
  
- Pre-process data
  - Bandpass filter
  - Channel selection
  
- Epoch data
  - Data is cropped to a specific time window
  
- Balance dataset
  - Unbalanced groups in the dataset are balanced to avoid bias during training
  
All the data processing pipeline is done step by step, recording the output of one step and using the stored data as input to the next step. This way, if the user wants to change some parameter in the middle of the pipeline, there is no need to run all the processes again which might take some time.

In *Data inspection*, the user can verify the quality of the processed dataset before continuing with model training in the next section.

### Data structure

All files are stored as Pickle files with the following datastructures and types:

The data is stored as *numpy.ndarray* with the following structure:

* epoched_data:
  * Type: *numpy.ndarray*
  * Dimensions: \[Trial, Channel, Time sample\]

* epoched_data_labels:
  * Type: *dict*
  * Fields:
    * 'fb_windowOnset' 0
    * 'fb_windowOnsetSamples': 0
    * 'fb_windowSize': 600
    * 'fb_windowSizeSamples': 307

* balanced_data

* balanced_data_labels

* filtered_metadata:
* epoched_metadata:
* balanced_metadata:

## Deep Learning Model

This section uses [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) as the Machine Learning library and [Comet ML](https://www.comet.ml/) to visualize the results of the trained model.

1. Generate a **Comet experiment**
    * Associate a Comet API key to the current project and name the current experiment (optional)
      * The Comet Api Key can be easily accessed after signing up to [Comet](https://www.comet.ml/site/) with Github or mail address. Project and Workspace are optional but help organize which experiments belong to which modeling projects.
    * The API key (together with the project name and workspace name) are loaded from a file present in the root directory of the Drive. If none is present run the code which is commented in the previous cell after editing the necessary fields
    * A link to the Live Comet experiment will be provided after cell run
    * While executing *trainer.fit(model_name)* in the **Train** section, head to the Comet UI in the browser to visualize the training: metrics, parameters, code, system metrics, and more, all in real time.
2. Define **hyperparameters**
    * Besides the basic hyperparameters, more can be addded.
    * These are passed to the Comet Logger.
3. Download **dataset**
    * After running cell, choose the appropriate file and the dataset will be uploaded.
4. Define **Model**
    * The model architecture is defined in the class at the **Define Model** section.
    * It uses the **LightningModule** from [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to abstract and separate the Research/Science component from the Engineering component.
    * Layers (convolution, max pooling, drop-outs, ...) are defined under the Model class constructer (*\_\_init\_\_*) and its feeding pipeline is defined in the *forward* function.
    * Download and split dataset
    * DataLoaders...
    * Oprimizer and Scheduler
    *
5. Terminate Experience
    * In a Jupyter or Colab notebook, Experiences are ended by running *comet_logger.experiment.end()*.

---

# Licence

Repository under MIT license.

# BibTeX
If you want to cite the repository feel free to use this:

```bibtex
@mastersthesis{correia2020,
  author       = {Correia, JR}, 
  title        = {Feedback classification in Brain-Computer interfaces using Convolutional Neural Networks},
  school       = {Instituto Superior Tecnico},
  year         = {2020},
}
```

