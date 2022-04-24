from os import listdir
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import random
import pickle
from torchvision import transforms
import re
import json

class PETDataset(Dataset):

    def __init__(self, 
                dir_path, names, 
                divided = False,
                augmentations = False,
                part = 1
                ):
        """
        Args:
            dir_path (string): Path to the main directory
        """
        self.dir_path = dir_path
        self.divided = divided
        self.augmentations = augmentations
        self.names = names
        self.part = part
        with open('names_dict.json', 'r') as f:
            self.names_d = json.loads(f.read())


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.dir_path + '/' + self.names[idx] + '.npy'
        
        #1 - pet, 0 - ct
        standard_transorms = transforms.Compose([
                          transforms.Resize(140),
                          transforms.CenterCrop(112),

                          ])
  
        #transforms.Normalize((0.485),(0.229))            

        images0 = torch.from_numpy(np.load(img_name)[:, 0, ...]).unsqueeze(0)
        #.unsqueeze(0)

        images1 = torch.from_numpy(np.load(img_name)[:, 1, ...]).unsqueeze(0)

        images = torch.cat((images0, images1))
        
        n1 = int((images.shape[1])*0.3*(self.part - 1))
        n2 = int((images.shape[1])*0.2*self.part + (images.shape[1])*0.2)
        images = images[:, n1:n2, ...]/255.
        
        if images.shape[1] < 60:
            shape_pad = (images.shape[0], 60 - images.shape[1], images.shape[2], images.shape[3])
            images = torch.hstack([images, torch.zeros(shape_pad)])
        if images.shape[1] > 60:
            images = images[:, :60, ...]

        
        images = standard_transorms(images)
             
        
        titles = {1: ['Органы малого таза:', 'Костная система:'],
                  2: ['Органы брюшной полости:', 'Органы малого таза:'],
                  3: ['Органы грудной клетки:', 'Органы брюшной полости:'],
                  4: ['{}.*?{}'.format('Область головы', 'шеи:'), 'Органы грудной клетки:']}
        
        first_title = titles[self.part][0]
        last_title = titles[self.part][1]
        # first_title = 'Диагноз по МКБ-10:'
        # last_title = 'Заключение:'       

        pat = re.compile('{}(.*){}'.format(first_title, last_title))
        text_path = self.dir_path + '/' + self.names[idx] + '.txt.txt'
        with open(text_path) as f:
            text = f.read().rstrip().replace('\n', ' ')
            text = re.sub( r'[\(\)]', '', text).replace('  ', ' ')
            sent = pat.findall(text)
            text_part = ''.join(sent).strip()
            text_part = re.sub(r'[^\w\s\.]', '',text_part)
        
        sample = {'image': images, 'text': text_part, 'name': int(self.names_d[self.names[idx]])}

        return sample

