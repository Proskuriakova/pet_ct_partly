import torchmetrics
import numpy as np
import torch
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('sberbank-ai/ruRoberta-large')

def collate_fn(data):

    texts = [data[i]['text'] for i in range(len(data))]
    images = [data[i]['image'] for i in range(len(data))]
    names = torch.Tensor([data[i]['name'] for i in range(len(data))])
    images = torch.stack([data[i]['image'] for i in range(len(data))])
    #images_min_shape = np.min([data[i]['image'].shape[1] for i in range(len(data))])
    #images = torch.stack([images[i][:, :images_min_shape, ...] for i in range(len(images))])
    #names = torch.stack([data[i]['name'] for i in range(len(data))])
    text_encoded = tokenizer(texts, return_tensors="pt", padding = True, truncation = True)
    ids = text_encoded['input_ids']
    masks = text_encoded['attention_mask']
    data = {'text_ids': ids, 'text_masks': masks, 'images': images, 'names': names}
#     max_l_txt = np.max([len(texts[i]) for i in range(len(texts))])
#     max_l_img = np.max([images[i].shape[3] for i in range(len(images))])


#     for i in range(len(data)):
#         img = data[i]['image']
#         added = torch.zeros(list(img.shape[:-1]) + [max_l_img - img.shape[-1]])
#         data[i]['image'] = torch.cat([data[i]['image'],added], dim = 3)

    return data


def replace_batchnorm_instance(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm3d):
            child: torch.nn.BatchNorm3d = child
            setattr(module, name, torch.nn.InstanceNorm3d(child.num_features))
        else:
            replace_batchnorm_instance(child)

def replace_batchnorm_group(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm3d):
            child: torch.nn.BatchNorm3d = child
            setattr(module, name, torch.nn.GroupNorm(1, child.num_features))
        else:
            replace_batchnorm_group(child)


