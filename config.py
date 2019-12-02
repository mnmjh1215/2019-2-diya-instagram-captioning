# all configs are saved here
import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder
    encoded_size = 14  # size of encoded image
    encoder_finetune = False
    # decoder
    encoder_dim = 1024
    decoder_dim = 512
    attention_dim = 512
    dropout = 0.5
    embed_dim = 300
    embedding_finetune = True

    lr = 5e-4

    # dataloader
    batch_size = 32
    num_workers = 4
    max_text_length = 30
    max_hashtag_length = 25
    
    # generating vocab
    min_text_freq = 1
    min_hashtag_freq = 2

    # train
    num_epochs = 30
    log_every = 1
    validation_freq = 1
    patience = 5


class ShowAttConfig(Config):
    checkpoint_path = "checkpoint/showatt.pth"


class OursConfig(Config):
    checkpoint_path = "checkpoint/ours.pth"
