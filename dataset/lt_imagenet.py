from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from .randaugment import RandAugmentMC
import torchvision

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label #, path

# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):

    txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    if phase not in ['train', 'val']:
        transform = data_transforms['test']
    else:
        transform = data_transforms[phase]

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, transform)

    if phase == 'test' and test_open:
        open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

rs = 256
cc = 224
# rs = 32
# cc = 32

transform_labeled = transforms.Compose([
        transforms.Resize(rs),
        transforms.CenterCrop(cc),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_val = transforms.Compose([
        transforms.Resize(rs),
        transforms.CenterCrop(cc),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(rs),
            transforms.CenterCrop(cc),
            transforms.RandomHorizontalFlip(),
            ])
        self.strong = transforms.Compose([
            transforms.Resize(rs),
            transforms.CenterCrop(cc),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def get_lt_imagenet(args):
    train_labeled_dataset = LT_Dataset(args.path_to_imagenet,  \
                    f'./IN_LT_splits/{args.labeled_perc}percent_labeled_new.txt',
                    transform=transform_labeled)
    train_unlabeled_dataset = LT_Dataset(args.path_to_imagenet, \
                    f'./IN_LT_splits/ImageNet_LT_train.txt',
                    transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_dataset = LT_Dataset(args.path_to_imagenet, \
                    f'./IN_LT_splits/ImageNet_LT_test.txt',
                    transform=transform_val)
    print(f"Labeled: {len(train_labeled_dataset)} | unlabeled: {len(train_unlabeled_dataset)} | test: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_imagenet(args):

    train_labeled_dataset = LT_Dataset(args.path_to_imagenet,  \
                    f'./IN_splits/{args.labeled_perc}percent_labeled.txt',
                    transform=transform_labeled)
    train_unlabeled_dataset = torchvision.datasets.ImageFolder(args.path_to_imagenet + '/train',
                    transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_dataset = torchvision.datasets.ImageFolder(args.path_to_imagenet + '/val',
                    transform=transform_val)
    print(f"Labeled: {len(train_labeled_dataset)} | unlabeled: {len(train_unlabeled_dataset)} | test: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_imagenet_100k(args):

    train_labeled_dataset = LT_Dataset(args.path_to_imagenet,  \
                    f'./IN_splits/100k_labeled.txt',
                    transform=transform_labeled)
    train_unlabeled_dataset = torchvision.datasets.ImageFolder(args.path_to_imagenet + '/train',
                    transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_dataset = torchvision.datasets.ImageFolder(args.path_to_imagenet + '/val',
                    transform=transform_val)
    print(f"Labeled: {len(train_labeled_dataset)} | unlabeled: {len(train_unlabeled_dataset)} | test: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset