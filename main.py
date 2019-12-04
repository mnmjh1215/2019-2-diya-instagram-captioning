

from models.show_att import Encoder, Decoder
from models.ours import ResNextEncoder, LookBackDecoder
from train import Trainer
from dataloader import get_dataloader
from config import Config
from utils import generate_vocab, OktDetokenizer, load_model, load_pretrained_embedding, tokenize_fn
from konlpy.tag import Okt

import json
import os
import argparse

JSON_FILES = {
    'train': 'data/DGU-Dataset/kts/train.json',
    'val': 'data/DGU-Dataset/kts/val.json',
    'test': 'data/DGU-Dataset/kts/test.json',
    'total': 'data/DGU-Dataset/kts/total.json'
}

def main(args):
    
    # train
    if args.mode == 'train':
        
        if args.vocab_file:
            with open(args.vocab_file) as fr:
                vocab = json.load(fr)
        else:
            # Create vocabulary if not given
            print("Creating vocabulary")
            train_json = JSON_FILES['train']
            targets = []
            with open(train_json) as fr:
                for item in json.load(fr):
                    if args.target_type == 'hashtag':
                        targets.extend(item['hashtag'])
                    else:
                        tokens = tokenize_fn(item['text'])
                        targets.extend(tokens)
            vocab = generate_vocab(targets, args.vocab_min_freq)
                
            with open(args.target_type + "_vocab_{0}.json".format(args.vocab_min_freq), "w") as fw:
                json.dump(vocab, fw)
        
        # prepare dataloader
        print("Loading DataLoader")
        train_dataloader = get_dataloader(JSON_FILES['train'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                          batch_size=Config.batch_size, num_workers=Config.num_workers, on_ram=args.load_image_on_ram)
        val_dataloader = get_dataloader(JSON_FILES['val'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                        batch_size=1, num_workers=Config.num_workers, on_ram=args.load_image_on_ram,
                                        shuffle=False)
        
        # prepare model
        print("Loading Model")
        
        if args.model == 'showatt':
            encoder = Encoder(Config.encoded_size)
            decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        elif args.model == 'ours':
            encoder = ResNextEncoder(Config.encoded_size)
            decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        elif args.model == 'ablation_resnext':
            encoder = ResNextEncoder(Config.encoded_size)
            decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        else:
            # ablation_lookback
            encoder = Encoder(Config.encoded_size)
            decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
            
        if args.target_type == 'text':
            # load pretrained embedding
            decoder.load_embedding(load_pretrained_embedding(vocab).to(Config.device))
        
        # prepare trainer
        trainer = Trainer(encoder, decoder, train_dataloader, val_dataloader, target_type=args.target_type, lr=args.lr)
        if args.checkpoint_load_path:
            # load checkpint
            trainer.load(args.checkpoint_load_path)
            
        # train!
        print("Start Training using device {0}".format(Config.device))      
        if not os.path.isdir(os.path.dirname(args.checkpoint_save_path)):
            os.makedirs(os.path.dirname(args.checkpoint_save_path))
        trainer.train(args.num_epochs, args.checkpoint_save_path)
        
        
    # test
    else:
        # TODO: 모델이 주어지면 해당 모델로 테스트셋에 inference를 실시하고, 여러 metric을 사용하여 결과 측정.
        pass



def get_args():
    parser = argparse.ArgumentParser("Train or test Show Attend and Tell model using instagram caption & hashtag data")

    parser.add_argument('model',
                        default='showatt',
                        choices=['showatt', 'ours', 'ablation_resnext', 'ablation_lb'],
                        help="model to use, one of showatt, ours, ablation_resnext, ablation_lb")
    
    parser.add_argument('mode',
                        default='train',
                        choices=['train', 'test'],
                        help="train or test mode")
    
    parser.add_argument('target_type',
                        default='text',
                        choices=['text', 'hashtag'],
                        help="target, text or hashtag")
    
    parser.add_argument('--checkpoint_load_path',
                        default=None,
                        help="checkpoint for either training or testing. Required for testing")
    
    parser.add_argument('--checkpoint_save_path',
                        default=Config.checkpoint_path,
                        help="path to save checkpoint while training")
    
    parser.add_argument('--vocab_file',
                        default=None,
                        help="vocab file path, must be json.")
    
    parser.add_argument('--vocab_min_freq',
                        default=1,
                        type=int,
                        help="minimum frequency of word/hashtag to be included in vocabulary")
    
    parser.add_argument('--num_epochs',
                        default=Config.num_epochs,
                        type=int,
                        help="number of epochs to train")
    
    parser.add_argument('--lr',
                        default=Config.lr,
                        type=float,
                        help="learning rate")
    
    parser.add_argument('--load_image_on_ram',
                        default=False,
                        action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)