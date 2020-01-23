

from models.show_att import Encoder, Decoder
from models.resnext_lb import ResNextEncoder, LookBackDecoder
from train import Trainer
from dataloader import get_dataloader
from config import Config
from utils import generate_vocab, OktDetokenizer, load_model, load_pretrained_embedding, tokenize_fn
from metric import avg_f1_score, avg_bleu, avg_rouge_l, avg_meteor
from konlpy.tag import Okt
import torch

from tqdm import tqdm

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
                
            if not os.path.isdir('vocabs/'):
                os.mkdir('vocabs/')
            with open("vocabs/" + args.target_type + "_vocab_{0}.json".format(args.vocab_min_freq), "w") as fw:
                json.dump(vocab, fw)
        
        # prepare dataloader
        print("Loading DataLoader")
        train_dataloader = get_dataloader(JSON_FILES['train'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                          batch_size=args.batch_size, num_workers=Config.num_workers, load_on_ram=args.load_image_on_ram)
        val_dataloader = get_dataloader(JSON_FILES['val'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                        batch_size=1, num_workers=Config.num_workers, load_on_ram=args.load_image_on_ram,
                                        shuffle=False)
        
        # prepare model
        print("Loading Model")
        print(args.model)
        
        if args.model == 'showatt':
            encoder = Encoder(Config.encoded_size)
            decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        elif args.model == 'resnext_lb':
            encoder = ResNextEncoder(Config.encoded_size)
            decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        elif args.model == 'resnext':
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
        if not os.path.isdir('checkpoint/'):
            os.makedirs('checkpoint/')
            
        checkpoint_save_path = "checkpoint/{0}_{1}_{2}.pth".format(args.model, args.target_type, args.vocab_min_freq)
        trainer.train(args.num_epochs, checkpoint_save_path)
        
        
    # test
    else:
        assert args.vocab_file is not None
        assert args.checkpoint_load_path is not None
        
        print("Loading vocab...")
        with open(args.vocab_file) as fr:
            vocab = json.load(fr)
            
        print("Loading model...")
        print(args.model)
        if args.model == 'showatt':
            encoder = Encoder(Config.encoded_size)
            decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        elif args.model == 'resnext_lb':
            encoder = ResNextEncoder(Config.encoded_size)
            decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        elif args.model == 'resnext':
            encoder = ResNextEncoder(Config.encoded_size)
            decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        else:
            # lookback
            encoder = Encoder(Config.encoded_size)
            decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(vocab))
        
        encoder = encoder.to(Config.device)
        decoder = decoder.to(Config.device)
        
        load_model(encoder, decoder, args.checkpoint_load_path)
        
        encoder.eval()
        decoder.eval()
        
        test_dataloader = get_dataloader(JSON_FILES['test'], vocab, type=args.target_type, tokenize_fn=tokenize_fn,
                                        batch_size=1, num_workers=Config.num_workers, load_on_ram=args.load_image_on_ram,
                                        shuffle=False)
        
        print("Running test...")
        if args.target_type == 'hashtag':
            f1, prec, rec = test_hashtag(encoder, decoder, test_dataloader, vocab)
            print("avg F1: {0:.4f}".format(f1))
            print('avg Precision: {0:.4f}'.format(prec))
            print('avg Recall: {0:.4f}'.format(rec))
        elif args.target_type == 'text':
            bleu1, rouge_l, meteor = test_text(encoder, decoder, test_dataloader, vocab)
            print('avg BLEU-1: {0:.4f}'.format(bleu1))
            print('avg ROUGE-L: {0:.4f}'.format(rouge_l))
            print('avg METEOR: {0:.4f}'.format(meteor))


def test_text(encoder, decoder, test_dataloader, vocab):
    # text의 각 metric의 결과를 리턴
    actuals = test_dataloader.dataset.tokens
    preds = []
    idx2word = dict([(v, k) for k, v in vocab.items()])
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for ix, (image, target, length) in tqdm(enumerate(test_dataloader)):
            image = image.to(Config.device)
            target = target.to(Config.device)
            
            encoded_image = encoder(image)
            prediction, alphas = decoder.generate_caption_greedily(encoded_image, 
                                                                   test_dataloader.dataset.vocab['<start>'],
                                                                   test_dataloader.dataset.vocab['<end>'])
            
            target = target[0, 1:-1]
            target = target.tolist()
            
            prediction = prediction[1:-1]
            prediction = [idx2word[idx] for idx in prediction]
            
            actuals.append(target)
            preds.append(prediction)
    
    print("Calculating BLEU-1")
    avg_bleu1_score = avg_bleu(actuals, preds, n=1)
    
    print("Calculating ROUGE-L")
    avg_rouge_l_score = avg_rouge_l(actuals, preds)
    
    print("Calculating METEOR")
    avg_meteor_score = avg_meteor(actuals, preds)
    
    return avg_bleu1_score, avg_rouge_l_score, avg_meteor_score


def test_hashtag(encoder, decoder, test_dataloader, vocab):
    # F1 제대로 계산한거 맞나...?
    actuals = test_dataloader.dataset.tokens
    preds = []
    idx2word = dict([(v, k) for k, v in vocab.items()])
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for ix, (image, target, length) in tqdm(enumerate(test_dataloader)):
            image = image.to(Config.device)
            target = target.to(Config.device)
            
            encoded_image = encoder(image)
            prediction, alphas = decoder.generate_caption_greedily(encoded_image, 
                                                                   test_dataloader.dataset.vocab['<start>'],
                                                                   test_dataloader.dataset.vocab['<end>'])
            
            target = target[0, 1:-1]
            target = target.tolist()
            
            prediction = prediction[1:-1]
            prediction = [idx2word[idx] for idx in prediction]
            
            preds.append(prediction)
    
    avg_f1, avg_prec, avg_rec = avg_f1_score(actuals, preds)
    
    return avg_f1, avg_prec, avg_rec


def get_args():
    parser = argparse.ArgumentParser("Train or test model using instagram caption & hashtag data")

    parser.add_argument('model',
                        default='showatt',
                        choices=['showatt', 'resnext_lb', 'resnext', 'lb'],
                        help="model to use, one of showatt, resnext_lb, resnext, lb")
    
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
    
    parser.add_argument('--batch_size',
                        default=Config.batch_size,
                        type=int,
                        help="batch size to be used while training")
    
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