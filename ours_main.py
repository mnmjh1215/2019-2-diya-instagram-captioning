

from models.ours import ResNextEncoder, LookBackDecoder
from train import Trainer
from dataloader import get_dataloader
from config import ShowAttConfig as Config
from utils import generate_vocab, OktDetokenizer, load_model
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
        okt = Okt()
        def tokenize_fn(text):
            tokens = okt.pos(text, norm=True, join=True)
            return tokens
        
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
            vocab = generate_vocab(targets, 1)
            with open(args.target_type + "_vocab.json", "w") as fw:
                json.dump(vocab, fw)
        
        # prepare dataloader
        print("Loading DataLoader")
        train_dataloader = get_dataloader(JSON_FILES['train'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                          batch_size=Config.batch_size, num_workers=Config.num_workers, on_ram=args.load_image_on_ram)
        val_dataloader = get_dataloader(JSON_FILES['val'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                        batch_size=Config.batch_size, num_workers=Config.num_workers, on_ram=args.load_image_on_ram,
                                        shuffle=False)
        
        # prepare model
        print("Loading Model")
        encoder = ResNextEncoder(Config.encoded_size)
        decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        
        # prepare trainer
        trainer = Trainer(encoder, decoder, train_dataloader, val_dataloader, Config)
        if args.checkpoint_load_path:
            # load checkpint
            trainer.load(args.checkpoint_load_path)
            
        # train!
        print("Start Training")
        if not os.path.isdir(os.path.dirname(args.checkpoint_save_path)):
            os.makedirs(os.path.dirname(args.checkpoint_save_path))
        trainer.train(args.num_epochs, args.checkpoint_save_path)
        
        
    # test
    else:
        # TODO: 모델이 주어지면 해당 모델로 테스트셋에 inference를 실시하고, 여러 metric을 사용하여 결과 측정.
        pass



def get_args():
    parser = argparse.ArgumentParser("Train or test Ours model using instagram caption & hashtag data")
    
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
    
    parser.add_argument('--num_epochs',
                        default=Config.num_epochs,
                        help="number of epochs to train")
    
    parser.add_argument('--load_image_on_ram',
                        default=False,
                        action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)