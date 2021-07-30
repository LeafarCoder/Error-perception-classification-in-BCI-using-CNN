# Error perception classification in Brain-Computer Interfaces using CNN

Master thesis by Rafael Correia (2020)

Contact me by email: [jose.r.c.correia@tecnico.ulisboa.pt](jose.r.c.correia@tecnico.ulisboa.pt)

More about me:
* [LinkedIn](https://www.linkedin.com/in/joserafaelcorreia/)
* [Sourcerer](https://sourcerer.io/leafarcoder)

---

# Contents
* [Overview](#overview)
* [Initial setup](#initial-setup)
* [Run experiments](#run-experiments)
  - [Setup](#setup)
  - [Process dataset](#process-dataset)
  - [Deep Learning Model](#deep-learning-model)
* [CNN Models](#cnn-models)
* [Results in Comet ML](#results-in-comet-ml)
* [Licence](#licence)
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
<details>
  <summary>Click to expand!</summary>
  
To get started follow this steps:

1. Download the [notebook](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/BCI_Feedback_Classifier.ipynb) and upload it into a [Google Drive](https://drive.google.com/).
2. In the Google Drive create a new folder to be the root directory of all the datasets, models and necessary resources for the project (as an example let's call this folder *BCI_root*).
3. Access [Google Colab](https://colab.research.google.com/) and login with the same Google account used for Google Drive.
   * Open a notebook (File > Open notebook OR Ctrl+O) and choose the BCI_Feedback_Classifier.ipynb notebook.
   * On the left Summary menu (if you can't see it select View > Summary) go to *1.4 Define project directories*.
   * Define ```running_online ``` as ```True``` (if it is not already).
   * Define ```project_root_folder``` as the root directory in your Google Drive as defined before with the prefix ```/content/drive/My Drive/``` (following the example set above this would be ```project_root_folder = "/content/drive/My Drive/BCI_root/"```).
4. In the Google Drive, inside the root folder (*/BCI_root* in our example) add 3 folders: *Datasets/* (to keep the raw and pre-processed datasets), *Models/* (to keep the CNN models), and *Images/* (optional; to keep images of pre-processed data or others needed).
   * Read the commented section in *1.4 Define project directories* to complete and better understand the tree structure of the directory.
   * The directory structure can be changed keeping the rest of the code intact by altering the references to the folders in the end of section *1.4 Define project directories*.
5. Inside */BCI_root/Datasets/* add as many folders as the different datasets you want to have with the proper naming for the dataset such as */BCI_root/Datasets/Dataset_1/*
   * Inside each dataset create 2 other folders: *all/* to place all the raw ".csv" files and *Datasets_pickle_files/* where all the pre-processing intermediate steps will be stored.
   * Finally, inside *Datasets_pickle_files/* add as many folders as the different levels of processing you intend to perform. This thesis uses 4: *Raw/*, *Pre-processed/*, *Epoched/* and *Balanced/*.

Concerning the dataset used for training, you can use your own or the one used in this thesis.

### Monitoring error-related potentials
The dataset used in this thesis is in this [page](http://bnci-horizon-2020.eu/database/data-sets) under the name *Monitoring error-related potentials (013-2015)*. The header *Data* is followed by 12 files (6 subjects with 2 sessions each).

1. Download all the 12 files locally.
2. Run this [Matlab file]() to convert the files from Matlab variable file (.mat extension) to Comma Separated Values file (.csv extension).
   * Change the load and save directories inside the Matlab function as necessary.
   * After the program runs, you should have one files called *AllLabels.csv* and 120 other .csv files (10 runs for each subject and session) named with the format "Subject\<sub\>\_s\<sess\>\_run\<r\>.csv".
3. Copy these files to the Google Drive
   * Place *AllLabels.csv* in the folder *BCI_root/Datasets/Monitoring_error-related_potentials_2015/*
   * Place all the other 120 files in the folder *BCI_root/Datasets/Monitoring_error-related_potentials_2015/all/* ("all" just refers to the fact that both test and train example are present in the dataset).

### New dataset

To use a new dataset some substantial change might be needed, depending on the format of the original data. To minimize altering the code downstream, try to keep the changes before the dataset processing step, forcing the dataset to have the datatype required by the following steps.

</details>

## Offline mode (desktop)

<details>
  <summary>Click to expand!</summary>
  
Same steps as in [Online mode](#online-mode) but without the need of using Google Drive. Just store all the files locally in your machine (with the same structure as presented above).

</details>

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

<details>
  <summary>Click to expand!</summary>
  
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

</details>

---

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

| Author(s) (Year) | Model name | Target ERP | Original paper | PyTorch (Lightning) model |
|---------------|------------|------------|:------------:|:--------------------:|
| Cecotti *et al.* (2011) | [CCNN](#ccnn) | P300 | [Paper](https://pubmed.ncbi.nlm.nih.gov/20567055/) | [CCNN](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/CCNN.py) |
| Manor *et al.* (2015) | [CNN-R](#cnn-r)| P300 | [Paper](http://journal.frontiersin.org/Article/10.3389/fncom.2015.00146/abstract) | [CNN-R](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/CNN-R.py) |
| Liu *et al.* (2017) | [BN3](#bn3) | P300 | [Paper](https://doi.org/10.1016/j.neucom.2017.08.039) | [BN3](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/BN3.py) |
| Shan *et al.* (2018) | [OCLNN](#oclnn) | P300 | [Paper](https://www.ijcai.org/proceedings/2018/222) |  |
| Luo *et al.* (2018) | [CNN-L](#cnn-l) | ErrP | [Paper](https://ieeexplore.ieee.org/document/8621183/) | [CNN-L](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/CNN-Luo.py) |
| Torres *et al.* (2018) | [ConvNet](#convnet) | ErrP | [Paper](https://ieeexplore.ieee.org/document/8512183/) | [ConvNet](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvNet.py) |
| Bellary *et al.* (2019) | [ConvArch](#convarch) | ErrP | [Paper](https://ieeexplore.ieee.org/document/8776901/) | [ConvArch1](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch1.py)<br/>[ConvArch1_Adam](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch1_Adam.py)<br/>[ConvArch2](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch2.py)<br/>[ConvArch2_BN](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch2_BN.py)<br/>[ConvArch2_Dropout](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch2_Dropout.py) |

Access all the models [here](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/tree/master/Models).

## CCNN

* Original [paper](https://pubmed.ncbi.nlm.nih.gov/20567055/)
   * H. Cecotti and A. Gräser, “Convolutional neural networks for P300 detection with application to brain-computer interfaces,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 33, no. 3, pp. 433–445, **2011**.
* Target ERP
   * **P300**
* Modifications
   * ...
* PyTorch [model](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/CCNN.py).

## CNN-R

* Original [paper](http://journal.frontiersin.org/Article/10.3389/fncom.2015.00146/abstract)
   * R. Manor and A. B. Geva, “Convolutional Neural Network for Multi-Category Rapid Serial Visual Presentation BCI,” Front. Comput. Neurosci., vol. 9, no. DEC, pp. 1–12, Dec. **2015**.
* Target ERP
   * **P300**
* Modifications
   * ...
* PyTorch [model](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/CCN-R.py).

## BN3

* Original [paper](https://doi.org/10.1016/j.neucom.2017.08.039)
   * M. Liu, W. Wu, Z. Gu, Z. Yu, F. F. Qi, and Y. Li, “Deep learning based on Batch Normalization for P300 signal detection,” Neurocomputing, vol. 275, pp. 288–297, **2017**.
* Target ERP
   * **P300**
* Modifications
   * ...
* PyTorch [model](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/BN3.py).

## OCLNN

* Original [paper](https://www.ijcai.org/proceedings/2018/222)
   * H. Shan, Y. Liu, and T. Stefanov, “A simple convolutional neural network for accurate P300 detection and character spelling in brain computer interface,” IJCAI Int. Jt. Conf. Artif. Intell., pp. 1604–1610, **2018**.
* Target ERP
   * **P300**
* Modifications
   * ...
* PyTorch [model](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/OCLNN.py).

## CNN-L

* Original [paper](https://ieeexplore.ieee.org/document/8621183/)
   * T. Luo, Y. Fan, J. Lv, and C. Zhou, “Deep reinforcement learning from error-related potentials via an EEG-based brain-computer interface,” in 2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), **2018**, pp. 697–701.
* Target ERP
   * **ErrP**
* Modifications
   * ...
* PyTorch [model](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/CNN-Luo.py).

<img src="https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/readme_img/_CNN-L_model.png" height="200">

## ConvNet

* Original [paper](https://ieeexplore.ieee.org/document/8512183/)
   * J. M. M. Torres, T. Clarkson, E. A. Stepanov, C. C. Luhmann, M. D. Lerner, and G. Riccardi, “Enhanced Error Decoding from Error-Related Potentials using Convolutional Neural Networks,” Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. EMBS, vol. 2018-July, pp. 360–363, **2018**.
* Target ERP
   * **ErrP**
* Modifications
   * ...
* PyTorch [model](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvNet.py).

<img src="https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/readme_img/_ConvNet_model.png" height="200">

## ConvArch

* Original [paper](https://ieeexplore.ieee.org/document/8776901/)
   * S. A. Swamy Bellary and J. M. Conrad, “Classification of Error Related Potentials using Convolutional Neural Networks,” in 2019 9th International Conference on Cloud Computing, Data Science & Engineering (Confluence), **2019**, pp. 245–249.
* Target ERP
   * **ErrP**
* Modifications
   * ...
* PyTorch models
   * [ConvArch1](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch1.py)
   * [ConvArch1 (using Adam instead of SGD)](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch1_Adam.py)
   * [ConvArch2](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch2.py)
   * [ConvArch2 (only BatchNorm applied)](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch2_BN.py)
   * [ConvArch2 (only Dropout applied)](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/Models/ConvArch2_Dropout.py)

<img src="https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/readme_img/_ConvArch_model.png" height="200">

---

# Results in Comet ML

Accessing [Comet ML](https://www.comet.ml/) allows the visualization of the training process and the performance metrics defined for the CNN models.

This image shows the loss and accuracy of the validation set during training:

![](https://github.com/LeafarCoder/BCI-Feedback-Classification-using-CNN/blob/master/readme_img/comet_1.JPG)

Comet has multiple tabs for each experiment that allows to organize the information from the experiment in a very nice way.

<details>
  <summary>Click to see all tabs in Comet!</summary>
  
* Charts
* Panels
* Code
* Hyper Parameters
* Metrics
* Graph Definition
* Output
* System Metrics
* Installed Packages
* Notes
* Graphics
* Audio
* Text
* Confusion Matrices
* Other
* HTML
* Assets

</details>

---

# Licence

Repository under [MIT licence](https://github.com/LeafarCoder/Error-perception-classification-in-BCI-using-CNN/blob/master/LICENSE).

# BibTeX
If you want to cite the repository feel free to use this:

```bibtex
@mastersthesis{correia2020,
  author       = {Correia, JR}, 
  title        = {Error perception classification in Brain-Computer interfaces using Convolutional Neural Networks},
  school       = {Instituto Superior Tecnico},
  year         = {2020},
}
```

