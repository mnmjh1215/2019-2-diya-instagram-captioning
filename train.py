# defines training procedure

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config


# TODO


class Trainer:
    """
    Trainer for encoder and decoder
    """

    def __init__(self, encoder, decoder, dataloader, config):
        """
        encoder: Encoder module
        decoder: Decoder module
        dataloader: instance of torch.utils.data.DataLoader
        config: Config instance containing learning rates and other hyperparameters
        """
        # TODO
        self.config = config
        self.device = self.config.device

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.dataloader = dataloader

        encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
        decoder_params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
        trainable_params = encoder_params + decoder_params
        
        self.optimizer = optim.Adam(params=trainable_params, lr=config.lr)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
        self.loss_hist = []
        self.curr_epoch = 0
        
        self.log_freq = config.log_freq
        
    def train(self, num_epochs):
        for epoch in range(self.curr_epoch, num_epochs):
            epoch_loss = 0

            start = time.time()
            for ix, (images, targets, lengths) in enumerate(self.dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                loss = self.train_step(images, targets, lengths)
                epoch_loss += loss
                
                if (ix + 1) % self.log_freq == 0:
                    print("[{0}/{1}] loss: {2}".format(epoch+1, num_epochs, epoch_loss / (ix + 1)))


            # end of epoch
            print("epoch {0} {1:.4f} seconds, loss: {2:.4f}".format(epoch + 1, time.time() - start,
                                                               epoch_loss / (ix + 1)))
                  
            self.curr_epoch += 1
            
    def train_step(self, images, targets, lengths):
        
        self.optimizer.zero_grad()
        
        encoded_images = self.encoder(images)
        predictions, alphas = self.decoder(encoded_images, targets)
        predictions = predictions.transpose(1, 2)  # (batch_size, length, C) -> (batch_size, C, length)
        
        targets = targets[:, 1:]
        mask = (targets > 0).type_as(targets)
        
        ce_loss = self.criterion(predictions, targets)
        ce_loss = ((ce_loss * mask) / mask.sum()).sum()
        
        ce_loss.backward()
        
        self.optimizer.step()
        self.loss_hist.append(ce_loss.item())
        
        return ce_loss.item()
        
        

            
    
