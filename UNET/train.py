import copy
import time
from collections import defaultdict
from lossfunction import *
import torch
from tqdm import tqdm

"""
define train model. 
Input: dataloader, model, optimizier, scheduler(for learning rate), and epochs
Output: the best model
Every epoch, there is a training step and a validating step. 
Update the best model when validating loss is smaller than best loss.
"""

# If cuda is available, use cuda. Otherwise, use cpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(dataloaders, model, optimizer, scheduler, num_epochs=25):
    # deep copy the current model weights as the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # recode the starting time
        since = time.time()

        # There is a training and validating step, respectively, each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()   # update learning rate every 10 epochs
                for param_group in optimizer.param_groups: 
                    print("LR", param_group['lr'])
                model.train()   # set model to training mode
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).permute(0, 3, 1, 2).float()   # change shape to (batch size, 3, 256, 512)

                # set the parameter gradient to be zero
                optimizer.zero_grad()

                # forward and track history only when phase is train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss, metrics = calc_loss(outputs, labels, metrics)

                    # backward + optimize only when phase is train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples   # calculate the mean of loss

            # update the best model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
       
        # calculate the time for every epoch
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model





