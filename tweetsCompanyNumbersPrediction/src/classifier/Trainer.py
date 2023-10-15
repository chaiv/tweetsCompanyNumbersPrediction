'''
Created on 20.04.2023

@author: vital
'''
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from classifier.transformer.DatasetUtils import createDataloader
from lightning_fabric.accelerators.cuda import find_usable_cuda_devices

class Trainer(object):

    def train(self,batch_size, epochs, num_workers, pad_token_idx, model, train_data, val_data, test_data, loggerPath, loggerName, checkpointPath, checkpointName
              ,accelerator='cuda', devices=find_usable_cuda_devices(1)
              ):
        train_loader = createDataloader(train_data, batch_size, num_workers, pad_token_idx)
        val_loader = createDataloader(val_data, batch_size, num_workers, pad_token_idx)
        test_loader = createDataloader(test_data, batch_size, num_workers, pad_token_idx)
        logger = TensorBoardLogger(
            save_dir=loggerPath, 
            name=loggerName)
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss", 
            mode="min", 
            dirpath=checkpointPath, 
            filename=checkpointName, 
            save_weights_only=True)
        early_stopping = EarlyStopping(monitor="valid_loss", mode="min", patience=10)
        trainer = pl.Trainer(
            max_epochs=epochs, 
            accelerator=accelerator, 
            devices=devices, 
            logger=logger, 
            callbacks=[checkpoint_callback, early_stopping], 
            accumulate_grad_batches=1)
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, dataloaders=test_loader)
        