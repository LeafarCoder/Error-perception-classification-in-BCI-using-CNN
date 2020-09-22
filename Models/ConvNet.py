import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import random

class Net(pl.LightningModule):

  # +++++++++++++++++++++++++++ Explain architecture +++++++++++++++++++++++++++
  def explainModel():
    # Explain the model using HTML text
    # This will be added to the HTML tab in Comet
    text = ""

    # Model name
    model_name = "ConvNet"
    text += "<h1>{}</h1>".format(model_name)

    # Model key points
    key_points =   """
    <p>This model is taken from the paper:<br>
    <i>J. M. M. Torres, T. Clarkson, E. A. Stepanov, C. C. Luhmann, M. D. Lerner, and G. Riccardi, “Enhanced Error Decoding from Error-Related Potentials using Convolutional Neural Networks,” Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. EMBS, vol. 2018-July, pp. 360–363, 2018.</i></p>

    <p>It defines a 4 layers CNN architecture.</p>
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

  def __init__(self, train_dataset, val_dataset, test_dataset, hyperparams={}):
    super(Net, self).__init__()
    
    self.test_dataset = test_dataset
    self.val_dataset = val_dataset
    self.train_dataset = train_dataset
    
    # Define hyperparameters (use the values defined in the paper)
    if(hyperparams == {}):
      # Default hyper-parameters from original work
      self.hyper_params = {
        'input_size': test_dataset[0][0].shape,
        'num_classes': 2,
        'batch_size': 120,               # Batch size (try powers of 2)
        'test_batch_size': 1,
        'max_num_epochs': 1200,
        'optimizer': 'SGD',              # SGD, Adam, ...
        'learning_rate': 0.0001,          # Learning rate for Optimizer
        # 'momentum': 0.9,                 # Momentum for Optimizer
        'weight_decay': 0.002,          # Weight decay (L2 regularization) 
      }
    else:
      self.hyper_params = hyperparams

    # **************** Declare layers **************** 
    self.layer1 = nn.Sequential(
        nn.Conv2d(1,128,kernel_size=(2,20)),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=(1,5), stride=(1,2)),
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(128,64,kernel_size=(1,10)),
        nn.ELU(),
        #nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),
    )
    self.layer3 = nn.Sequential(
        nn.Flatten(),           # Flattens
        nn.Linear(64*12,2),     # Fully Connected layer
        # To apply Cross Entropy Loss don't apply Softmax (it's included there)
    )

    # +++++++++++++++++++++++++ Loss function +++++++++++++++++++++++++
    self.loss_function = nn.CrossEntropyLoss()

  def forward(self, x):
    # Convert input to float (match network weights type)
    x = x.float()

    # Convert from [Batch, Channel, Length] to [Batch, Channel, Height, Width]
    # Do this in order to correctly apply 2D convolution
    # Get current size format
    s = x.shape
    # Convert
    x = x.view(s[0], 1, s[1], s[2])

    # Crop width of signal to 64 at random point (as described in the paper)
    r = random.randint(1,512-64)
    x = x[:,:,:,r:(r+64)]

    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    # apply Linear layer
    out3 = self.layer3(out2)
  
    # print("\nSizes:\n{}\n{}\n{}\n{}\n".format(
    #   x.shape,
    #   out1.shape,
    #   out2.shape,
    #   out3.shape,
    # ))

    # Return output
    return out3

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # +++++++++++++++++++ Test, Validation and Test steps +++++++++++++++++++++++
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # ---------- Train ----------
  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      y = y[:,4] # get only label
      y_logits = self.forward(x)

      loss = self.loss_function(y_logits, y)
      y_hat = torch.round(F.softmax(y_logits, dim=-1))
      acc = self.binary_acc(y_hat, y)
      
      # Define outputs
      # result = pl.TrainResult(loss)
      # result.log('loss_train', loss, on_step=True, on_epoch=False, logger=True, prog_bar=False, reduce_fx=torch.mean)
      # result.log('acc_train', acc.clone().detach(), on_step=True, on_epoch=False, logger=True, prog_bar=False, reduce_fx=torch.mean)
      # return result
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
      # print("Val y_hat: batch {}\n{}".format(batch_idx, [y_hat_ss[0:5], y_hat[0:5]]))

      loss = self.loss_function(y_logits, y)
      y_hat = torch.round(F.softmax(y_logits, dim=-1))
      acc = self.binary_acc(y_hat, y)

      # Define outputs
      output = {
        'loss_val': loss,
        'acc_val': acc.clone().detach()
        }

      # print("Output val: {}".format(output))

      # result = pl.EvalResult()
      # result.log('loss_val', loss)
      # result.log('acc_val', acc.clone().detach())
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
    y_hat = torch.round(F.softmax(y_logits, dim=-1))
    acc = self.binary_acc(y_hat, y)

    # Specificy the subject from where this sample comes from:
    subj = y_all[:,0].tolist()
    subj_str = 'acc_' + str(subj[0])

    # A meio do teste posso fazer logs. Ex do log de uma imagem:
    #self.logger.experiment.add_image('example_images', grid, 0)
    # Fazer log dos gráficos usados para teste (um ou vários canais)

    # Define outputs
    output = {
      'acc': acc.clone().detach(),
      subj_str: acc.clone().detach(),
      'y_true': y.clone().detach(),
      'y_predicted': torch.tensor([int(x[0]==0) for x in y_hat])    # convert list of pairs to y_prediction
    }
    return output

  def test_epoch_end(self, outputs):
    print(outputs[0])
    
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

  # +++++++++++++++++++++++++++ Get data and split (test, val, train) +++++++++++++++++++++++++++

  def prepare_data(self):
    # All datasets are downloaded passed to the Model Constructor.
    # No further preparation needed.
    pass

  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++ Data Loaders +++++++++++++++++++++++++++
  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.hyper_params['batch_size'], num_workers=8, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.hyper_params['batch_size'], num_workers=8, shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.hyper_params['test_batch_size'], num_workers=8)

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++ Oprimizer and Scheduler +++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  def configure_optimizers(self):
    # Use Stochastic Gradient Discent
    optimizer = torch.optim.SGD(  self.parameters(),
                                  lr = self.hyper_params['learning_rate'],
                                  weight_decay = self.hyper_params['weight_decay'])
    
    #scheduler = StepLR(optimizer, step_size=1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # return [optimizer], [scheduler]
    return [optimizer]

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # ++++-+++++++++++++++++++++++++++ Auxiliar functions +++++++++++++++++++++++++++
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  # Measure binary accuracy
  def binary_acc(self, y_hat, y):
    correct = torch.tensor([y_hat[idx,y[idx]] for idx in range(len(y))])

    correct_results_sum = correct.sum()
    acc = correct_results_sum/len(y)
    acc = torch.round(acc * 100)

    return acc

  # Return test y_true and y_predicted vectors for Confusion Matrix in Comet ML
  def get_test_labels_predictions(self):
    return (self.test_y_true, self.test_y_predicted)
