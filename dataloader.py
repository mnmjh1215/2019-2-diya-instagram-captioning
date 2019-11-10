# for some reason, torchvision.datasets.Flickr8k doesn't work on my data,
# which was downloaded from https://www.kaggle.com/srbhshinde/flickr8k-sau
# so I made my own dataloader

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json

from config import Config


target_size = 224
default_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])


class DGUDataset(Dataset):
    """
    Custom dataset designed to load DGU Dataset using either one of ['total.json', 'train.json', 'val.json', 'test.json']
    """
    def __init__(self, json_file, vocab, transform=default_transform, type='hashtag', tokenize_fn=None):
        """
        json_file: path to json file
        vocab: dictionary object. must include <PAD>, <UNK>, <start>, <end> as its keys.
        transform: transform to be applied to image
        type: 'hashtag' or 'caption'
        tokenize_fn: function that is used to tokenize text when type is 'caption'
        """
        assert any(json_file.endswith(file_type) for file_type in ['total.json', 'train.json', 'val.json', 'test.json'])
        assert type in ['hashtag', 'caption']
        if type == 'caption':
            assert tokenize_fn is not None
        with open(json_file) as fr:
            self.json = json.load(fr)
        self.root_dir = '/'.join(json_file.split('/')[:-1])
        self.type = type
        self.vocab = vocab
        self.transform = transform
        self.tokenize_fn = tokenize_fn
                
    def __getitem__(self, index):
        item = self.json[index]
        
        # load target = hashtag or caption
        if self.type == 'hashtag':
            hashtags = item['hashtag']
            # 해시태그의 경우, vocab에 존재하지 않으면 그냥 무시
            target = [self.vocab.get(hashtag) for hashtag in hashtags if hashtag in self.vocab]
        elif self.type == 'caption':
            text = item['text']
            tokens = self.tokenize_fn(text)
            UNK_idx = self.vocab['<UNK>']
            target = [self.vocab.get(token, UNK_idx) for token in tokens]
        target = [self.vocab.get('<start>')] + target + [self.vocab.get('<end>')]
        target = torch.LongTensor(target)
        
        # load image
        image = Image.open(os.path.join(self.root_dir, item['image_path'])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target
    
    def __len__(self):
        return len(self.json)
    
    
def collate_fn(data):
    """
    data: list of tuple (image, target)
    
    returns:
        images: torch tensor of shape (batch_size, 3, img_height, img_width)
        targets: torch tensor of shape (batch_size, padded_length)
        lengths: list of 'real' length of each target
    """
    
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, targets = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge targets (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in targets]
    targets = torch.zeros(len(targets), max(lengths)).long()
    for i, cap in enumerate(targets):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def get_dataloader(json_file, vocab, transform=default_transform, type='hashtag', tokenize_fn=None,
                   batch_size=Config.batch_size, shuffle=True, num_workers=-1):
    
    dataset = DGUDataset(json_file, vocab, transform=transform, type=type, tokenize_fn=tokenize_fn)
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return loader
        