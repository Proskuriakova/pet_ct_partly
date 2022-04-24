import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

from model import ModelPET
import torch.nn.functional as F
from loss import NTXentLoss

from dataset import PETDataset
from utils import collate_fn, Emb_Save
from os import listdir
from collections import OrderedDict

import logging as log
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AutoModel
from transformers import RobertaTokenizer, RobertaModel
from mmcv.runner import get_dist_info

import warnings
warnings.filterwarnings("ignore")


class PET_Model(pl.LightningModule):
    
    class DataModule(pl.LightningDataModule):
        def __init__(self, model_instance):
            super().__init__()
            self.save_hyperparameters(model_instance.hparams)
            
            self.names = []
            for dir_content in listdir(self.hparams.path_to_data):
                if dir_content.split('.')[-1] == 'npy':
                    self.names.append(dir_content.split('.')[0])

            num_train = len(self.names)
            print("LEN", num_train)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.hparams.valid_size * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]
            
            self.train_names = [self.names[i] for i in train_idx]
            self.valid_names = [self.names[i] for i in valid_idx]

        def train_dataloader(self) -> DataLoader:
            
            self.train_dataset = PETDataset(dir_path = self.hparams.path_to_data, names = self.names,
                                            divided = self.hparams.divided_text, part = self.hparams.part, 
                                            augmentations = self.hparams.augmentations)
 
            return DataLoader(
                dataset = self.train_dataset,
                drop_last = True, sampler = DistributedSampler(self.train_dataset),
                batch_size = self.hparams.batch_size, num_workers = self.hparams.loader_workers,
                collate_fn = collate_fn
            )



        def test_dataloader(self) -> DataLoader:
            
            self.test_dataset = PETDataset(dir_path = self.hparams.path_to_data, names = self.names,
                                            divided = self.hparams.divided_text, part = self.hparams.part,
                                              augmentations = self.hparams.augmentations)
             
            return DataLoader(
                dataset=self.test_dataset, sampler = DistributedSampler(self.test_dataset),
                batch_size = self.hparams.batch_size, num_workers = self.hparams.loader_workers,
                drop_last = True,collate_fn = collate_fn
            )

    def __init__(self, hparams: Namespace) -> None:
        super(PET_Model, self).__init__()
        # save_hyperparameters https://discuss.pytorch.org/t/pytorch-lightning-module-cant-set-attribute-error/121125/5
        self.save_hyperparameters(hparams)
        print(self.hparams)
 
        self.image_embeds = []
        self.text_embeds = []
        self.embed_file_name = self.hparams.embeds_file
        
        self.names = []
        
        self.batch_size = self.hparams.batch_size
        self.out_dim = self.hparams.out_dim
        self.tokenizer = RobertaTokenizer.from_pretrained(self.hparams.text_encoder_model)

        self.data = self.DataModule(self)

        self.__build_model()

        self.__build_loss()


    def __build_model(self) -> None:
        self.model_pet = ModelPET(self.hparams.image_encoder_model, self.hparams.text_encoder_model,
                             self.hparams.out_dim, self.hparams.bucket_size, self.hparams.rolling_window,
                             self.hparams.freeze_layers, self.hparams.divided_text, self.hparams.pretrained,
                             self.hparams.norm, self.hparams.num_ch)

        
    def __build_loss(self):
        #data = torch.Tensor(2)
        #self._loss = nn.MSELoss()
        self._loss = NTXentLoss(self.hparams.temperature,
                               self.hparams.use_cosine_similarity, self.hparams.alpha_weight)
     
    def forward(self, xis, xls_ids, xls_masks, mode):
        return self.model_pet(xis, xls_ids, xls_masks, mode)
        

    def loss(self, text_embed, image_embed) -> torch.tensor:

        return self._loss(text_embed, image_embed)

    def predict_step(self, batch: tuple, batch_nb: int) -> list:
        xls = batch['texts']
        xis = batch['images']
        name = batch['names']

        zis, zls = self.model_pet(xis, xls)
        
        return [zls.cpu().numpy(), zis.cpu().numpy(), name]

    
    def sync_tensor_across_gpus(self, t
                                ) :
        # t needs to have dim 0 for troch.cat below. 
        # if not, you need to prepare it.
        if t is None:
            return None
        group = dist.group.WORLD
        group_size = torch.distributed.get_world_size(group)
        gather_t_tensor = [torch.zeros_like(t) for _ in
                           range(group_size)]
        dist.all_gather(gather_t_tensor, t) 
        return torch.cat(gather_t_tensor, dim=0)
    
            
    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        xis = batch['images']
        xls_ids = batch['text_ids']
        xls_masks = batch['text_masks']
        names = batch['names']
        
        # xls_ids = batch['input_ids']
        # xls_masks = batch['input_masks']
        #print('xis', xis)
        zis, zls = self.forward(xis, xls_ids, xls_masks, mode = 'train')
        loss_val = self._loss(zis, zls)
        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)      

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )        

        return output

    
    def test_step(self, batch: list, batch_nb: int, *args, **kwargs):
        self.model_pet.eval()
        with torch.no_grad():          
            xis = batch['images']
            xls_ids = batch['text_ids']
            xls_masks = batch['text_masks']
            names = batch['names']

            zis, zls = self.forward(xis, xls_ids, xls_masks, mode = 'test')
            zis_all = self.sync_tensor_across_gpus(zis)
            zls_all = self.sync_tensor_across_gpus(zis)
            names_all = self.sync_tensor_across_gpus(names)
            test_loss = self._loss(zis, zls)            
            self.image_embeds.extend(zis_all.cpu().numpy())
            self.text_embeds.extend(zls_all.cpu().numpy())
            self.names.extend(names_all.cpu().numpy())        
        
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)     
         
        tqdm_dict = {"test_loss": test_loss}
        
        output = OrderedDict({"test_loss": test_loss, 
                             "progress_bar": tqdm_dict, "log": tqdm_dict})

        return output
    

    
    def test_step_end(self, output):
        
        img_name = 'results/image_embeddings_' + self.embed_file_name + '.npy'
        txt_name = 'results/text_embeddings_' + self.embed_file_name + '.npy'
        f_name = 'results/names_' + self.embed_file_name + '.txt'
        
        texts_embeds = np.array(self.text_embeds)
        with open(txt_name, 'wb') as f:
            np.save(f, texts_embeds)
        images_embeds = np.array(self.image_embeds)
        with open(img_name, 'wb') as f:
            np.save(f, images_embeds)  
        with open(f_name, 'w') as f:
            for item in self.names:
                f.write("%s\n" % item)
    
    
#     def test_epoch_end(self):
        
#         self.save_emdeds.compute(self.hparams.out_name)
        
#         print('ALL', len(self.test_texts_embeds))
#         print('zero element', len(self.test_texts_embeds[0]))
#         texts_embeds = np.array(self.test_texts_embeds)
#         with open('texts_embeddings_bs3.npy', 'wb') as f:
#             np.save(f, texts_embeds)
#         images_embeds = np.array(self.test_images_embeds)
#         with open('images_embeddings_bs3.npy', 'wb') as f:
#             np.save(f, images_embeds)
                 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model_pet.parameters(), lr = self.hparams['learning_rate'])
        return [optimizer], []

     
    @classmethod
    def add_model_specific_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        
        parser.add_argument(
            "--path_to_data",
            default = "/data/burenko/datasets/pet-ct",
            type = str,
            help = "Path to the data",
        )
        parser.add_argument(
            "--text_encoder_model",
            default = 'sberbank-ai/ruRoberta-large',
            type = str,
            help = "Text encoder",
        )
        parser.add_argument(
            "--image_encoder_model",
            default = "resnet18_3D",
            type = str,
            help = "Image encoder",
        )        
        parser.add_argument(
            "--temperature",
            default = 1e-02,
            type = float,
            help = "Temperature for loss calculation",
        )
        parser.add_argument(
            "--learning_rate",
            default = 1e-05,
            type = float,
            help = "learning rate",
        )
        parser.add_argument(
            "--valid_size",
            default = 0.2,
            type = float,
            help = "Size of validation sample",
        )        
        parser.add_argument(
            "--use_cosine_similarity",
            default = True,
            type = bool,
            help = "Using cosine similarity in loss or not - bool, default True",
        )
        parser.add_argument(
            "--alpha_weight",
            default = 0.5,
            type = int,
            help = "Loss parameter, default = 0.75",
        )

        parser.add_argument(
            "--bucket_size",
            default = 32,
            type = np.int64,
            help = "Count of images processing per time, default - 32,"\
            "if None - all images per patient processing together (need big GPU)",
        )
        parser.add_argument(
            "--rolling_window",
            default = 8,
            type = np.int64,
            help = "Rolling window per images",
        )        
        parser.add_argument(
            "--out_dim",
            default = 300,
            type = np.int64,
            help = "Size of output embeddings, default - 300",
        )
        parser.add_argument(
            "--freeze_layers",
            default = [0, 1, 2, 3],
            type = list,
            help = "",
        )        
        parser.add_argument(
            "--loader_workers",
            default = 1,
            type = int,
            help = "Count of workers",
        )
        parser.add_argument(
            "--divided_text",
            default = True,
            type = bool,
            help = "Divide text into logical parts or not, bool, default - True",
        )        

        parser.add_argument(
            "--out_name",
            default = 'tmp',
            type = str,
            help = "Name for output files",
        )      
        parser.add_argument(
            "--augmentations",
            default = False,
            type = bool,
            help = "Name for output files",
        )    
        parser.add_argument(
            "--part",
            default = 3,
            type = int,
            help = "Part of body",
        )
        parser.add_argument(
            "--embeds_file",
            default = '2ch_gr',
            type = str,
            help = "Name of output files",
        )
        parser.add_argument(
            "--pretrained",
            default = False,
            type = bool,
            help = "Pretrained or not image encoder",
        )
        parser.add_argument(
            "--norm",
            default = 'group',
            type = str,
            help = "Normalization in image encoder",
        )     
        parser.add_argument(
            "--num_ch",
            default = 2,
            type = int,
            help = "Input number of channels",
        )             
        return parser