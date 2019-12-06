# defines training procedure

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config

from metric import avg_bleu, avg_f1_score


class Trainer:
    """
    Trainer for encoder and decoder
    """

    def __init__(self, encoder, decoder, dataloader, val_dataloader, target_type='text',
                 lr=Config.lr, log_every=Config.log_every, validation_freq=Config.validation_freq):
        """
        encoder: Encoder module
        decoder: Decoder module
        dataloader: instance of torch.utils.data.DataLoader
        config: config containing learning rates and other hyperparameters
        """
        # TODO
        self.device = Config.device

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        assert val_dataloader.batch_size == 1

        encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
        decoder_params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
        trainable_params = encoder_params + decoder_params
        
        self.optimizer = optim.Adam(params=trainable_params, lr=lr)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
        self.loss_hist = []
        self.curr_epoch = 0
        
        self.log_every = log_every
        self.validation_freq = validation_freq
        
        self.target_type = target_type
        
        self.best_val_score = 0
        
        
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
                val_score = self.validate()
                print("epoch {0}, validation score: {1:.4f}, {2:.4f} seconds".format(self.curr_epoch, val_score, time.time() - val_start))
                if val_score > self.best_val_score:
                    print("Best validation score, saving model...")
                    self.best_val_score = val_score
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
        
        actuals = []
        preds = []
        
        with torch.no_grad():
            for ix, (image, target, length) in enumerate(self.val_dataloader):
                image = image.to(self.device)
                target = target.to(self.device)
                
                encoded_image = self.encoder(image)
                prediction, alphas = self.decoder.generate_caption_greedily(encoded_image, 
                                                                                self.val_dataloader.dataset.vocab['<start>'],
                                                                                self.val_dataloader.dataset.vocab['<end>'])
                
                target = target[0, 1:-1]
                target = target.tolist()
                
                prediction = prediction[1:-1]
                
                actuals.append(target)
                preds.append(prediction)
                
        self.encoder.train()
        self.decoder.train()
        
        # if target type is text, calculate bleu-1
        if self.target_type == 'text':     
            bleu1 = avg_bleu(actuals, preds, n=1)
            
            return bleu1
        
        # if target type is hashtag, calculate f1
        elif self.target_type == 'hashtag':
            avg_f1 = avg_f1_score(actuals, preds)
            
            return avg_f1

    def save(self, savepath):
        torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.curr_epoch,
                'loss_hist': self.loss_hist,
                'best_val_score': self.best_val_score
            }, savepath)
        
    
    def load(self, loadpath):
        checkpoint = torch.load(loadpath)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curr_epoch = checkpoint['epoch']
        self.loss_hist = checkpoint['loss_hist']
        self.best_val_score = checkpoint['best_val_score']
