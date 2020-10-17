from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torch.utils.data import Dataset

import os
import os.path
import hashlib
import csv
import gzip
import errno
import tarfile
import zipfile
from io import BytesIO
import requests

import numpy as np

from tqdm.auto import tqdm, trange
from PIL import Image
import glob

from .util import download_and_extract_archive
# import transform

def cifar10_dataset(train_transforms, test_transforms):
    
    # train_transforms, test_transforms = transforms.cifar10_transforms()

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, 
    transform=train_transforms)

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
    transform=test_transforms)

    return trainset, testset


class TinyImageNet(Dataset):
    """Load Tiny ImageNet Dataset."""

    def __init__(self, path, train=True, train_split=0.7, download=True, random_seed=1, transform=None):
        """Initializes the dataset for loading.
        Args:
            path (str): Path where dataset will be downloaded.
            train (bool, optional): True for training data. (default: True)
            train_split (float, optional): Fraction of dataset to assign
                for training. (default: 0.7)
            download (bool, optional): If True, dataset will be downloaded.
                (default: True)
            random_seed (int, optional): Random seed value. This is required
                for splitting the data into training and validation datasets.
                (default: 1)
            transform (optional): Transformations to apply on the dataset.
                (default: None)
        """
        super(TinyImageNet, self).__init__()
        
        self.path = path
        self.train = train
        self.train_split = train_split
        self.transform = transform
        self._validate_params()

        # Download dataset
        if download:
            self.download()

        self._class_ids = self._get_class_map()
        self.data, self.targets = self._load_data()

        self._image_indices = np.arange(len(self.targets))

        np.random.seed(random_seed)
        np.random.shuffle(self._image_indices)

        split_idx = int(len(self._image_indices) * train_split)
        self._image_indices = self._image_indices[:split_idx] if train else self._image_indices[split_idx:]
    
    def __len__(self):
        """Returns length of the dataset."""
        return len(self._image_indices)
    
    def __getitem__(self, index):
        """Fetch an item from the dataset.
        Args:
            index (int): Index of the item to fetch.
        
        Returns:
            An image and its corresponding label.
        """
        image_index = self._image_indices[index]
        
        image = self.data[image_index]
        if not self.transform is None:
            image = self.transform(image)
        
        return image, self.targets[image_index]
    
    
    def _validate_params(self):
        """Validate input parameters."""
        if self.train_split > 1:
            raise ValueError('train_split must be less than 1')
    
    @property
    def classes(self):
        """List of classes present in the dataset."""
        return tuple(x[1]['name'] for x in sorted(
            self._class_ids.items(), key=lambda y: y[1]['id']
        ))
    
    def _get_class_map(self):
        """Create a mapping from class id to the class name."""
        with open(os.path.join(self.path, 'tiny-imagenet-200', 'wnids.txt')) as f:
            class_ids = {x[:-1]: '' for x in f.readlines()}
        
        with open(os.path.join(self.path, 'tiny-imagenet-200', 'words.txt')) as f:
            class_id = 0
            for line in csv.reader(f, delimiter='\t'):
                if line[0] in class_ids:
                    # class_ids[line[0]] = line[1].split(',')[0].lower()
                    class_ids[line[0]] = {
                        'name': line[1],
                        'id': class_id
                    }
                    class_id += 1
        
        return class_ids
    
    def _load_image(self, image_path):
        """Load an image from the dataset.
        Args:
            image_path (str): Path of the image.
        
        Returns:
            PIL object of the image.
        """
        image = Image.open(image_path)

        # Convert grayscale image to RGB
        if image.mode == 'L':
            image = np.array(image)
            image = np.stack((image,) * 3, axis=-1)
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        return image

    def _load_data(self):
        """Fetch data from each data directory and store them in a list."""
        data, targets = [], []

        # Fetch train dir images
        train_path = os.path.join(self.path, 'tiny-imagenet-200', 'train')
        for class_dir in os.listdir(train_path):
            train_images_path = os.path.join(train_path, class_dir, 'images')
            for image in os.listdir(train_images_path):
                if image.lower().endswith('.jpeg'):
                    data.append(
                        self._load_image(os.path.join(train_images_path, image))
                    )
                    targets.append(self._class_ids[class_dir]['id'])
        
        # Fetch val dir images
        val_path = os.path.join(self.path, 'tiny-imagenet-200', 'val')
        val_images_path = os.path.join(val_path, 'images')
        with open(os.path.join(val_path, 'val_annotations.txt')) as f:
            for line in csv.reader(f, delimiter='\t'):
                data.append(
                    self._load_image(os.path.join(val_images_path, line[0]))
                )
                targets.append(self._class_ids[line[1]]['id'])
        
        return data, targets
    
    def download(self):
        """Download the data if it does not exist."""
        print('Downloading dataset...')
        r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
        zip_ref = zipfile.ZipFile(BytesIO(r.content))
        zip_ref.extractall(os.path.dirname(self.path))
        zip_ref.close()

        # Move file to appropriate location
#         os.rename(
#             os.path.join(os.path.dirname(self.path), 'tiny-imagenet-200'),
#             self.path
#         )
        print('Done.')
        


# class TinyImageNet(Dataset):
#     url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
#     filename = 'tiny-imagenet-200.zip'
#     dataset_folder_name = 'tiny-imagenet-200'

#     EXTENSION = 'JPEG'
#     NUM_IMAGES_PER_CLASS = 500
#     CLASS_LIST_FILE = 'wnids.txt'
#     VAL_ANNOTATION_FILE = 'val_annotations.txt'

#     def __init__(self, root, train=True, train_split = 0.7, transform=None, target_transform=None, random_seed=1, download=False):
#         self.root = root
#         self.transform = transform
#         self.target_transform = target_transform

#         if download and (not os.path.isdir(os.path.join(self.root, self.dataset_folder_name))):
#             self.download()

#         self.split_dir = 'train' if train else 'val'
#         self.split_dir = os.path.join(
#             self.root, self.dataset_folder_name, self.split_dir)
#         self.image_paths = sorted(glob.iglob(os.path.join(
#             self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))

#         self.target = []
#         self.labels = {}

#         # build class label - number mapping
#         with open(os.path.join(self.root, self.dataset_folder_name, self.CLASS_LIST_FILE), 'r') as fp:
#             self.label_texts = sorted([text.strip()
#                                        for text in fp.readlines()])
#         self.label_text_to_number = {
#             text: i for i, text in enumerate(self.label_texts)}

#         # build labels for NUM_IMAGES_PER_CLASS images
#         if train:
#             for label_text, i in self.label_text_to_number.items():
#                 for cnt in range(self.NUM_IMAGES_PER_CLASS):
#                     self.labels[f'{label_text}_{cnt}.{self.EXTENSION}'] = i

#         # build the validation dataset
#         else:
#             with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
#                 for line in fp.readlines():
#                     terms = line.split('\t')
#                     file_name, label_text = terms[0], terms[1]
#                     self.labels[file_name] = self.label_text_to_number[label_text]

#         self.target = [self.labels[os.path.basename(
#             filename)] for filename in self.image_paths]

#         # added
#         self._image_indices = np.arange(len(self.target))

#         np.random.seed(random_seed)
#         np.random.shuffle(self._image_indices)

#         split_idx = int(len(self._image_indices) * train_split)
#         self._image_indices = self._image_indices[:split_idx] if train else self._image_indices[split_idx:]
    

        
#     def download(self):
#         download_and_extract_archive(
#             self.url, self.root, filename=self.filename)

#     def __getitem__(self, index):
#         filepath = self.image_paths[index]
#         img = Image.open(filepath)
#         img = img.convert("RGB")

#         image_index = self._image_indices[index] # added
#         target = self.target[image_index]

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self._image_indices) # added

