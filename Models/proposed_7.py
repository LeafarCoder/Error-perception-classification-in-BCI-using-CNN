import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

class Net(pl.LightningModule):

  # +++++++++++++++++++++++++++ Explain architecture +++++++++++++++++++++++++++
  def explainModel():
    # Explain the model using HTML text
    # This will be added to the HTML tab in Comet
    text = ""

    # Model name
    model_name = "_own10"
    text += "<h1>{}</h1>".format(model_name)

    # Model key points
    key_points =   """
    <p>This model is an experiment.</p>
    """
    # Further explain the architecture
    text += "{}".format(key_points)

    return text
  
  # Log hyperparameters in Comet
  def get_hyperparams(self):
    return self.hyper_params

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++ Define Architecture +++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  def __init__(self, input_size=None, hyperparams={}):
    super(Net, self).__init__()
    
    # FOR TRYING THE SGD OPTIMIZER!
    # Default hyper-parameters from original work
    self.hyper_params = {
      'input_size': input_size,
      'batch_size': 128,               # Batch size (try powers of 2)
      'test_batch_size': 1,
      'max_num_epochs': 500,
      'optimizer': 'SGD',              # SGD, Adam, ...
      'weight_decay': 1e-5,          # Weight decay (L2 regularization)
      'momentum': 0.9,                 # Momentum for Optimizer
      'learning_rate': 1e-3,          # Learning rate for Optimizer
    }

    # Overwrite hyperparameters if given
    if(hyperparams):
      for key,val in hyperparams.items():
        self.hyper_params[key] = val

    # **************** Declare layers **************** 
    self.layer0 = nn.BatchNorm2d(1)
    self.layer1 = nn.Sequential(
        nn.Conv2d(1,16,kernel_size=(1,20), stride=(1,20)),
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(16,16,kernel_size=(64,1)),
        nn.BatchNorm2d(16),
        nn.ELU(),
    )
    self.flatten = nn.Flatten()
    self.layer3 = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(240,1),
        # Sigmoid is not needed because it calculated is inside BCEWithLogitsLoss
        # If Sigmoid would be explicitly written here, then use BCELoss loss function.
        #nn.Sigmoid()
    )

    # +++++++++++++++++++++++++ Loss function +++++++++++++++++++++++++
    self.loss_function = nn.BCEWithLogitsLoss()

  def forward(self, x):
    # Convert input to float (match network weights type)
    x = x.float()

    # Convert from [Batch, Channel, Length] to [Batch, Channel, Height, Width]
    # Do this in order to correctly apply 2D convolution
    # Get current size format
    s = x.shape
    # Convert
    x = x.view(s[0], 1, s[1], s[2])

    # apply Batch Normalization to input
    out0 = self.layer0(x)
    out1 = self.layer1(out0)
    out2 = self.layer2(out1)
    out2_flatten = self.flatten(out2)
    out3 = self.layer3(out2_flatten)

    # Return output
    return out3

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # +++++++++++++++++++ Test, Validation and Test steps +++++++++++++++++++++++
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ---------- Train ----------
  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      y = y[:,4] # get only label

      # output: 1 node (logit. To classify first pass through sigmoid and use 0.5 as threshold [done in binary_acc])
      y_logits = self.forward(x)
      y_logits = y_logits.view(y_logits.shape[0])
      y = y.type_as(y_logits)

      loss = self.loss_function(y_logits, y)

      acc = self.binary_acc(y_logits, y)
      
      # Define outputs
      logs = {
        'loss_train': loss,
        'acc_train': acc.clone().detach()}
      output = {
        'loss_train': loss,
        'acc_train': acc.clone().detach(),
        'loss': loss,
        'log': logs
        }

      # print("Output train: {}".format(output))
      return output

  # ---------- Validation ----------
  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      y = y[:,4] # get only label
      y_logits = self.forward(x)
      y_logits = y_logits.view(y_logits.shape[0])
      y = y.type_as(y_logits)

      loss = self.loss_function(y_logits, y)

      acc = self.binary_acc(y_logits, y)

      # Define outputs
      output = {
        'loss_val': loss,
        'acc_val': acc.clone().detach()
        }

      # print("Output val: {}".format(output))
      return output

  def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['loss_val'] for x in outputs]).mean()
      avg_acc = torch.stack([x['acc_val'] for x in outputs]).mean()
      
      # Define outputs
      logs = {'loss_val': avg_loss, 'acc_val': avg_acc}
      return {'val_loss': avg_loss, 'log': logs}

  # ---------- Test ----------
  def test_step(self, batch, batch_idx):
    x, y_all = batch
    y = y_all[:,4] # get only label
    y_logits = self.forward(x)
    y_logits = y_logits.view(y_logits.shape[0])
    y = y.type_as(y_logits)

    loss = self.loss_function(y_logits, y)

    acc = self.binary_acc(y_logits, y)

    # Specificy the subject from where this sample comes from:
    subj = y_all[:,0].tolist()
    subj_str = 'acc_' + str(subj[0])

    # Calculate predicted outputs
    y_predicted = torch.sigmoid(y_logits)
    # define 0.5 as threshold
    y_predicted[y_predicted>=0.5] = 1.0
    y_predicted[y_predicted<0.5] = 0.0

    # Define outputs
    output = {
      'acc': acc.clone().detach(),
      subj_str: acc.clone().detach(),
      'y_true': y.clone().detach(),
      'y_predicted': y_predicted.clone().detach()    # convert list of pairs to y_prediction
    }
    return output

  def test_epoch_end(self, outputs):
    
    test_acc = torch.stack([x['acc'] for x in outputs]).mean()
    self.test_y_true = torch.stack([x['y_true'] for x in outputs])
    self.test_y_predicted = torch.stack([x['y_predicted'] for x in outputs])
    
    # Get accuracy per subject
    self.test_acc_subj = [0]*6
    for i in range(6):
      subj_str = 'acc_' + str(i+1)
      accuracies = [x[subj_str] for x in outputs if(subj_str in x)]
      if(not accuracies):   # if empty
        self.test_acc_subj[i] = 0
      else:
        self.test_acc_subj[i] = torch.stack(accuracies).mean()
    
    # Define outputs
    # General accuracy
    logs = {'acc_test': test_acc}
    # Accuracy per subject
    for i in range(6):
      subj_str = 'acc_subj_' + str(i+1)
      logs[subj_str] = self.test_acc_subj[i]

    #for i in range()
    return {'log': logs}

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++ Oprimizer and Scheduler +++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  def configure_optimizers(self):
    # Use Stochastic Gradient Discent
    optimizer = torch.optim.SGD(  self.parameters(),
                                  lr = self.hyper_params['learning_rate'],
                                  momentum = self.hyper_params['momentum'],
                                  weight_decay = self.hyper_params['weight_decay'])
    
    #scheduler = StepLR(optimizer, step_size=1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # return [optimizer], [scheduler]
    return [optimizer]

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # ++++-+++++++++++++++++++++++++++ Auxiliar functions +++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # Measure binary accuracy
  def binary_acc(self, y_logits, y):
    # apply sigmoid to output result (from single node)
    y_sigmoid = torch.sigmoid(y_logits)
    # define 0.5 as threshold
    y_sigmoid[y_sigmoid>=0.5] = 1
    y_sigmoid[y_sigmoid<0.5] = 0

    # Calculate accuracy (sum of all inputs equal to the targets divided by total number of targets)
    acc = 100 * (y_sigmoid == y).sum().type(torch.float) / len(y)

    return acc

  # Return test y_true and y_predicted vectors for Confusion Matrix in Comet ML
  def get_test_labels_predictions(self):
    return (self.test_y_true, self.test_y_predicted)
