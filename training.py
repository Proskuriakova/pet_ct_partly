import argparse
import os
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, CSVLogger
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
import numpy as np
from pet_model import PET_Model
from torch.utils.data import DataLoader, RandomSampler

import warnings
warnings.filterwarnings("ignore")

#запуск модели
def main(hparams) -> None:

#     # Инициализируем модель
    model = PET_Model(hparams)
    
    #  Модуль для раннего останова
    
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta = 0.01,
        patience = hparams.patience,
        verbose = True,
        mode = hparams.metric_mode,
    )

    # Логгер для тензорборда

    tb_logger = TensorBoardLogger(
        save_dir="./experiments",
        version="version_" + hparams.name_board,
        name="",
    )
    
    ckpt_path = os.path.join(
        "./checkpoints/", tb_logger.version,
    )
    
    csv_logger = CSVLogger('./', name = 'csv_logs', version = tb_logger.version)

    # Чекпоинты
    
    checkpoint_callback = ModelCheckpoint(
        verbose = True,
        dirpath = ckpt_path,
        filename = 'chk_' + hparams.name_board,
    )

    # Инициализация pl класса трэйнера
    #callbacks= early_stop_callback,

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        log_every_n_steps = 10,
        checkpoint_callback=True,
        gpus=[2, 3],
        log_gpu_memory="all",
        deterministic=True,
        fast_dev_run=False,
        precision = 16,
        strategy = 'ddp',
        auto_lr_find = True,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
    )
    #strategy="ddp",
    if hparams.phase == 'train':
    
#     #  запуск обучения
        print('START TRAINING')
        trainer.fit(model, model.data)
        trainer.save_checkpoint(hparams.name_board + ".ckpt")
        print('START TESTING')
        trainer.test(model, datamodule = model.data, verbose=True)
        
        #trainer.test(model, datamodule = model.data, verbose=True)
        #predictions = trainer.predict(model, datamodule=model.data)
#         predict_loader = model.data.predict_dataloader()
#         preds_img, preds_txt, names = [], [], []
#         trainer.model.eval()
#         trainer.model.to('cuda:1')
#         for i, batch in enumerate(predict_loader):
#             with torch.no_grad():
#                 xls = batch['texts']
#                 xis = [batch['images'][i].to('cuda:1')for i in range(len(batch['images']))]
#                 name = batch['names']
#                 zis, zls = trainer.model.model_pet(xis, xls, mode = 'predict')
#                 preds_img.extend(zis.cpu().numpy())
#                 preds_txt.extend(zls.cpu().numpy())
#                 names.extend(name)        
        

#         img_name = 'results/image_embeddings_' + hparams.file_name + '.npy'
#         txt_name = 'results/text_embeddings_' + hparams.file_name + '.npy'
#         f_name = 'results/names_' + hparams.file_name + '.txt'
        
#         texts_embeds = np.array(preds_txt)
#         with open(img_name, 'wb') as f:
#             np.save(f, texts_embeds)
#         images_embeds = np.array(preds_img)
#         with open(txt_name, 'wb') as f:
#             np.save(f, images_embeds)  
#         with open(f_name, 'w') as f:
#             for item in names:
#                 f.write("%s\n" % item)
    else:
    
        print('START TESTING')
        model = model.load_from_checkpoint('_csv_logs/version_bs2_chest_2ch_ddp_version_bs2_chest_2ch_ddp/checkpoints/epoch=199-step=9199.ckpt')
        trainer.test(model, datamodule = model.data, verbose=True)


#     chk_path = "./experiments/version_01-06-2021--18-09-58/checkpoints/epoch=2-step=315.ckpt"
#     model_saved = model.load_from_checkpoint(chk_path)
#    trainer.test(model, datamodule = model.data, verbose=True)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default = 2,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_loss", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default= "min",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=20,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )
    
    parser.add_argument(
        "--phase",
        default='train',
        type=str,
        help="the phase of work",
    )    

    # gpu args
    parser.add_argument("--gpus", type = list, default = [0, 3], help = "How many gpus")
    parser.add_argument(
        "--val_check_interval",
        default = 2.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    
    parser.add_argument(
        "--name_board",
        default='tmp',
        type=str,
        help=(
            "The name of tensorboard directory"
        ),
    )    
    parser.add_argument(
        "--file_name",
        default='tmp',
        type=str,
        help=(
            "The name of tensorboard directory"
        ),
    )  
    # each LightningModule defines arguments relevant to it
    parser = PET_Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)