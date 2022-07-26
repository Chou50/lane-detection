import os
import glob
from torch.utils.data import DataLoader
from dataload import *
from torchvision import transforms
import plotfigure
import torch
from torch import optim
import model
import train


DATA_PATH = '/Users/chou/kaggle/lane-detection-for-carla-driving-simulator-256/'

# train data
train_img = glob.glob(DATA_PATH + 'train/*.png')
train_img.sort()

# val data
val_img = glob.glob(DATA_PATH + 'val/*.png')
val_img.sort()

# label
train_label = [DATA_PATH + 'train_label/' + os.path.basename(x)[:-4] + '_label.png' for x in train_img]
val_label = [DATA_PATH + 'val_label/' + os.path.basename(x)[:-4] + '_label.png' for x in val_img]

# transformation for images.
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])   # Normalization is used for better convergence.

train_set = SimDataset(train_img, train_label, transform=trans)
val_set = SimDataset(val_img, val_label, transform=trans)

batch_size = 12
dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}


# plot image and image with mask to see the data
# images, masks = next(iter(dataloaders['train']))
# plotfigure.plot_figure(images[3], masks[3])


# start to train the model
# If cuda is available, use cuda. Otherwise, use cpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# semantic segmentation, 3 classifying outputs
model = model.SCNN(input_size=(512, 256)).to(device)


# use small learning rate here, otherwise, it is hard to converge
optimizer_ft = optim.Adam(model.parameters(), lr=5e-4)
# learning rate decreases 10% every 10 step.
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
model = train.train_model(dataloaders, model, optimizer_ft, exp_lr_scheduler, num_epochs=15)


# prediction
model.eval()

test_dataset = SimDataset(val_img, val_label, transform=trans)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)

inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()
pred = pred.transpose(0, 2, 3, 1)  # shape transposed to (batch size, H, W, 3)
pred = (pred > 0.5).astype(int)
print(pred.shape)

# plot the prediction
plotfigure.plot_figure(inputs[0], pred[0])
