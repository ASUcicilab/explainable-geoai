import copy
import os
import pickle
import random
import time

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# dataset class
class_names = ['crater', 'dunes', 'hill', 'icebergtongue', 'lake', 'meander', 'river', 'volcano']
cls_to_idx = {'crater': 0, 'dunes': 1, 'hill': 2, 'icebergtongue': 3, 'lake': 4, 'meander': 5, 'river': 6, 'volcano': 7}

class terrainDataset(Dataset):
    
    def __init__(self, sets='train', transform=None):
        self.dir = './datasets/natural_feature_dataset/'

        if sets == 'train':
            self.img_list = pickle.load(open(os.path.join(self.dir, 'train_set.pickle'), 'rb'))
        elif sets == 'test':
            self.img_list = pickle.load(open(os.path.join(self.dir, 'test_set.pickle'), 'rb'))

        self.transform = transform    
        random.shuffle(self.img_list)
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.dir, 'images', self.img_list[index])
        img_cls = self.img_list[index].split('_')[0]
        cls_idx = cls_to_idx[img_cls]
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'image': img, 'class': cls_idx, 'name': self.img_list[index]}
        
        return sample   


# intialize datasets and transformation 
transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_set = terrainDataset(sets='train', transform=transform)
test_set = terrainDataset(sets='test', transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
dataloader = {'train': train_loader, 'val': test_loader}
dataset_sizes = {'train': len(train_set), 'val': len(test_set)}


# model initialization 
model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 8)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)    


# loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 30

# train vgg16
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  
        else:
            model.eval()   

        running_loss = 0.0
        running_corrects = 0

        for sample in dataloader[phase]:
            inputs = sample['image']
            labels = sample['class']
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)    
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
torch.save(best_model_wts, './models/vgg16_best.pth')
