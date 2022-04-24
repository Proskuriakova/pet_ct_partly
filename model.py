import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from utils import replace_batchnorm_instance, replace_batchnorm_group
from transformers import AutoModel, AutoTokenizer,RobertaTokenizer, RobertaModel
from monai.networks.nets import HighResNet

import warnings
warnings.filterwarnings("ignore")

def pprint_snapshot():
    s = torch.cuda.memory_snapshot()
    for seg in s:
        print("%7.2f | %7.2f MB - %s" % (
            seg["active_size"] / 1000000., seg["total_size"] / 1000000., seg["segment_type"]))
        for b in seg["blocks"]:
            print("    %7.2f MB - %s" % (b["size"] / 1000000., b["state"]))

class ModelPET(nn.Module):
    def __init__(self, res_base_model, bert_base_model, out_dim, bucket_size, rolling_window,
                 freeze_layers, divided, pretrained, norm, num_ch):
        super(ModelPET, self).__init__()
        self.out_dim = out_dim
        self.bucket_size = bucket_size
        self.rolling_window = rolling_window
        self.divided = divided
        self.l_dim = int(320 / self.rolling_window)
        self.pretrained = pretrained
        self.norm = norm
        self.num_ch = num_ch
        
        #init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model,freeze_layers)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_base_model)
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(1024, 1024) #1024 is the size of the BERT embbedings (312 for tiny)
        self.bert_l2 = nn.Linear(1024, out_dim) #1024 is the size of the BERT embbedings
        
        # init Resnet
        self.res_base_model  = res_base_model 
        self.resnet_dict = {"resnet18_3D": models.video.r3d_18(pretrained = self.pretrained),
                            "resnet50": models.resnet50(pretrained=self.pretrained),
                            "resnet_2plus1": models.video.r2plus1d_18(pretrained = self.pretrained),
                           'monai_resnet': HighResNet(in_channels=1, norm_type='INSTANCE', out_channels = 300)}
        resnet = self._get_res_basemodel(res_base_model)
        self.res_base_model = res_base_model
        if res_base_model == 'monai_resnet':
            self.num_ftrs = 300
            self.res_features = resnet
            self.res_pooling  = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        else:
            self.num_ftrs = resnet.fc.in_features
            self.res_features = nn.Sequential(*list(resnet.children())[:-1])
            if self.norm == 'instance':
                replace_batchnorm_instance(self.res_features)
            if self.norm == 'group':
                replace_batchnorm_group(self.res_features)
        # # projection MLP for ResNet Model
        self.res_l1 = nn.Linear(self.num_ftrs, self.num_ftrs)
        self.res_l2 = nn.Linear(self.num_ftrs, self.out_dim)
        # #concat images projections per patient
        # self.im_l1_concat = nn.Linear(self.out_dim*self.l_dim, self.out_dim)
        # self.im_l2_concat = nn.Linear(self.out_dim, self.out_dim)
        # #concat text projections per patient
        # self.txt_l1_concat = nn.Linear(self.out_dim*7, self.out_dim)
        # self.txt_l2_concat = nn.Linear(self.out_dim, self.out_dim)
        
        
    def _get_res_basemodel(self, res_model_name):
        try:
            if res_model_name == 'resnet_2plus1':
                res_model = self.resnet_dict[res_model_name]
                # modify only the first conv layer
                origc = res_model.stem[0]  # the orig conv layer
                # build a new layer only with one input channel
                c1 = torch.nn.Conv3d(self.num_ch, origc.out_channels, kernel_size=origc.kernel_size,
                                     stride=origc.stride, padding=origc.padding, bias=origc.bias)
                # this is the nice part - init the new weights using the original ones
                # with torch.no_grad():
                #     c1.weight.data = origc.weight.data.sum(dim=2, keepdim=True)
                res_model.stem[0] = c1
                
            else:
                res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = RobertaModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("Text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def image_encoder(self, xis):
        h = self.res_features(xis)
        if self.res_base_model  == 'monai_resnet':
            h = self.res_pooling(h)        
        x = h.squeeze()
        x = self.res_l1(x)
        x = F.relu(x)
        x = self.res_l2(x)
 
        return x
    
        
    def text_encoder(self, encoded_inputs, mode):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        outputs = self.bert_model(**encoded_inputs)
        
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
        x = self.bert_l1(sentence_embeddings)
        x = F.relu(x)
        out_emb = self.bert_l2(x)

        return out_emb


    def text_encode(self, text, mode):
         x = self.text_encoder(text, mode)
        
        return x
    
  
            
    def forward(self, images_batch, text_ids, text_mask, mode):
        text_input = {'input_ids': text_ids, 'attention_mask': text_mask}
        zis = self.image_encoder(images_batch)
        zls = self.text_encode(text_input, mode)
        
        return zis.squeeze(), zls.squeeze()

    def text_extract(self, inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        encoded_inputs = self.tokenizer(inputs, 
                                         return_tensors="pt", 
                                         padding=True,
                                         truncation=True).to(next(self.bert_model.parameters()).device)
        outputs = self.bert_model(**encoded_inputs)
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
        return sentence_embeddings

        
    def extract_embeddings(self, images_batch, text_batch, mode):
        
        #zis = [self.image_encoder(images_batch[i].unsqueeze(0)).squeeze() for i in range(len(images_batch)) ]
        #zis_stack = torch.stack(zis, dim=0).float()        
        
        zls = self.text_encode(text_batch, mode)
        #zls_stack = torch.stack(zls, dim=0).float()

        
        zis = self.image_encoder(images_batch)
#         zis = [self.res_features(images_batch[i].unsqueeze(0)).squeeze() for i in range(len(images_batch)) ]
#         zis_stack = torch.stack(zis, dim=0).float()        
        
#         zls = [self.text_extract(text_batch[i]).squeeze() for i in range(len(text_batch)) ]
#         zls_stack = torch.stack(zls, dim=0).float()
       
        return zis.squeeze(), zls.squeeze()        