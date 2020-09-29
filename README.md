# BCI-Feedback-Classification-using-CNN
Master thesis by Rafael Correia (2020)

# Index
* [Overview](#overview)
* [Initial setup](#initial-setup)
* [Run experiments](#run-experiments)
  - [Setup](#setup)
  - [Process dataset](#process-dataset)
  - [Deep Learning Model](#deep-learning-model)
* [CNN Models](#cnn-models)
* [Results in Comet ML](#results-in-comet-ml)
* [License](#license)
* [BibTeX](#bibtex)

---
# Overview

The goal of this thesis is to classify the occurance of a feedback error, i.e., the perception of an error by a user interacting with a BCI (Brain-Computer Interface). The approach taken is to develop a Convolutional Neural Network (CNN) model.

CNNs are widely known in the image classification domain but they can also be used to classify other types of information such as Electroencephalography (EEG) which can be regarded as 2D data, if the spatial position of each channel is considered, or 1D data otherwise in which case each channel of the EEG gives a time-series signal and is considered individually.

The input given to the CNN for the classification is the EEG portion just after a feedback is presented. This can be the confirmation of a choice, the actual performance of an action or some other form of feedback. If the feedback is not the one expected (the performed action is different from the one intended, for example) then an Event-related potential (ERP) is elicited in the brain. Many different ERPs exist in response to different stimulus. The ERP specific for erroneous feedback is called Error Potential (ErrP) and happens around 300ms after the feedback is presented. Hence, givin the EEG signal just after the feedback to the CNN model, it is its task to classify it as presenting a ErrP or not.

Having this information in real time, i.e., knowing if the user realized an error in the system, allows the correction of the action or the improvement of the model controling thr BCI as depicted in the image below (taken from [here](https://www.frontiersin.org/articles/10.3389/fnins.2014.00208/full)).


![](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/readme_img/ErrP_to_improve_BCI.PNG)

---
# Initial setup

## Online mode

To get started follow this steps:

1. Download the [notebook](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/BCI_Feedback_Classifier.ipynb) and upload it into a [Google Drive](https://drive.google.com/).
2. In the Google Drive create a new folder to be the root directory of all the datasets, models and necessary resources for the project (as an example let's call this folder *BCI_root*).
3. Access [Google Colab](https://colab.research.google.com/) and login with the same Google account used for Google Drive.
   3.1. Open a notebook (File > Open notebook OR Ctrl+O) and choose the BCI_Feedback_Classifier.ipynb notebook.
   3.2. On the left Summary menu (if you can't see it select View > Summary) go to *1.4 Define project directories*.
   3.3. Define ```running_online ``` as ```True``` (if it is not already).
   3.4. Define ```project_root_folder``` as the root directory in your Google Drive as defined before with the prefix ```/content/drive/My Drive/``` (following the example set above this would be ```project_root_folder = "/content/drive/My Drive/BCI_root/"```).
4. In the Google Drive, inside the root folder (*/BCI_root* in our example) add 3 folders: *Datasets/* (to keep the raw and pre-processed datasets), *Models/* (to keep the CNN models), and *Images/* (optional; to keep images of pre-processed data or others needed).
   * Read the commented section in *1.4 Define project directories* to complete and better understand the tree structure of the directory.
   * The directory structure can be changed keeping the rest of the code intact by altering the references to the folders in the end of section *1.4 Define project directories*.

Open the notebook that was uploaded into the Google Drive with .
https://colab.research.google.com/notebooks/intro.ipynb


## Offline mode (desktop)

---

# Run experiments

The whole notebook document is organized in three main sections:
- [Setup](#setup)
- [Process dataset](#process-dataset)
- [Deep Learning Model](#deep-learning-model)


## Setup

In this section, the whole setup for the rest of the pre-processing and modeling is made.

First run *Necessary installs* to install all the required libraries to the Python environment in the remote Google Colab computer. Python will install the most recent updates on each package. If you notice one or more libraries are not interacting well with the rest then try downgrading the version by changing the code in *Necessary installs* like so:
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
  * Dimensions: \[#Trial, #Channel, Time sample\]

* epoched_data_labels:
  * Type: *numpy.ndarray*
  * Dimensions: \[#Trial, 5\]
  * Content per trial: \[*Subject*, *Session*, *Run*, *Trial*, *Label*\]
    * Label: **0** (negative feedback; error) or **1** (positive feedback; no error)

* balanced_data:
  * Type: ...

* balanced_data_labels
  * Type: ...

* filtered_metadata:

* epoched_metadata:
  * Type: *dict*
  * Fields:
    - *fb_windowOnset*: Start of epoching window after feedback in given number of miliseconds
    - *fb_windowOnsetSamples*: Start of epoching window after feedback in given number of time samples
    - *fb_windowSize*: Size of epoching window in given number of miliseconds
    - *fb_windowSizeSamples*: Size of epoching window in given number of time samples
    
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

# CNN Models

Access the models [here](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/tree/master/Models).

add sections with each of the models (CCNN, CNN-R, BN3, ...) with:
- Original paper
- Original purpose (P300, ErrP)
- How it was modified to ajust to the ErrP or database needs (ex: changing the kernel sizes and input sizes)
- Add images of the models with my abstract approach

# Results in Comet ML


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

