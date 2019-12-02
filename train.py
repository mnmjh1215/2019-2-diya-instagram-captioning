# defines training procedure

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    """
    Trainer for encoder and decoder
    """

    def __init__(self, encoder, decoder, dataloader, val_dataloader, config):
        """
        encoder: Encoder module
        decoder: Decoder module
        dataloader: instance of torch.utils.data.DataLoader
        config: config containing learning rates and other hyperparameters
        """
        # TODO
        self.config = config
        self.device = self.config.device

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
        decoder_params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
        trainable_params = encoder_params + decoder_params
        
        self.optimizer = optim.Adam(params=trainable_params, lr=config.lr)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
        self.loss_hist = []
        self.curr_epoch = 0
        
        self.log_every = config.log_every
        self.validation_freq = config.validation_freq
        
        self.best_val_loss = 10000
        
        
    def train(self, num_epochs, checkpoint_path):
        for epoch in range(self.curr_epoch, num_epochs):
            epoch_loss = 0

            start = time.time()
            for ix, (images, targets, lengths) in enumerate(self.dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                loss = self.train_step(images, targets, lengths)
                epoch_loss += loss
                
                if (ix + 1) % self.log_every == 0:
                    print("[{0}/{1}] loss: {2:.4f}, {3:.4f}".format(epoch+1, num_epochs, epoch_loss / (ix + 1), time.time() - start))

            # end of epoch
            print("epoch {0} {1:.4f} seconds, loss: {2:.4f}".format(epoch + 1, time.time() - start,
                                                               epoch_loss / (ix + 1)))
                  
            self.curr_epoch += 1
            
            if self.curr_epoch % self.validation_freq == 0:
                val_start = time.time()
                val_loss = self.validate()
                print("epoch {0}, validation loss: {1:.4f}, {2:.4f} seconds".format(self.curr_epoch, val_loss, time.time() - val_start))
                if val_loss < self.best_val_loss:
                    print("Best validation loss, saving model...")
                    self.best_val_loss = val_loss
                    self.save(checkpoint_path)
            
    def train_step(self, images, targets, lengths):
        
        self.optimizer.zero_grad()
        
        encoded_images = self.encoder(images)
        predictions, alphas = self.decoder(encoded_images, targets)
        predictions = predictions.transpose(1, 2)  # (batch_size, length, C) -> (batch_size, C, length)
        
        targets = targets[:, 1:]        
        targets = targets.to(self.device)
        
        ce_loss = self.criterion(predictions, targets)
        mask = (targets > 0).type_as(ce_loss)
        ce_loss = ((ce_loss * mask) / mask.sum()).sum()
        
        ce_loss.backward()
        
        self.optimizer.step()
        self.loss_hist.append(ce_loss.item())
        
        return ce_loss.item()
        
    def validate(self):
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        with torch.no_grad():
            for ix, (images, targets, lengths) in enumerate(self.val_dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                encoded_images = self.encoder(images)
                predictions, alphas = self.decoder(encoded_images, targets)
                predictions = predictions.transpose(1, 2)  # (batch_size, length, C) -> (batch_size, C, length)
                
                targets = targets[:, 1:]
                targets = targets.to(self.device)
                
                ce_loss = self.criterion(predictions, targets)
                mask = (targets > 0).type_as(ce_loss)
                ce_loss = ((ce_loss * mask) / mask.sum()).sum()
                
                total_loss += ce_loss.item()
                
        self.encoder.train()
        self.decoder.train()
        
        return total_loss / (ix + 1)

    def save(self, savepath):
        torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.curr_epoch,
                'loss_hist': self.loss_hist,
            }, savepath)
        
    
    def load(self, loadpath):
        checkpoint = torch.load(loadpath)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curr_epoch = checkpoint['epoch']
        self.loss_hist = checkpoint['loss_hist']
