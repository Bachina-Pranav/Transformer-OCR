import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchvision.utils import save_image
from src.ted import TextEncoderDecoder
import numpy as np

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

class PretrainingDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        super().__init__()
        if transform is None:
            self.transform = A.Compose([
                ToTensorV2()
            ])
        else:
            self.transform = transform
            
        self.root_dir = root_dir
        files = os.listdir(root_dir)
        self.images = [file for file in files if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png')]
        
    def __len__(self):
        return len(self.images) 
    
    def load_image(self, image_path):
        image = cv2.imread(f"{self.root_dir}/{image_path}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def __getitem__(self,idx):
        image = self.load_image(self.images[idx])
        image[image < 127] = 0
        image[image >= 127] = 255
        image = image/255
        if self.transform:
            image = self.transform(image=image)['image']
        return image
    
    
class PageData(Dataset):
    def __init__(self, split, path='/ssd_scratch/cvit/saigunda/iam'):
        assert split in ['train', 'val', 'test'], "Split should be one of ['train', 'val', 'test']"
        
        self.path = path
        self.split = split
        self.random_seed = 42
        
        self.p2a = self._load_annot()
        self.len_p2a = len(self.p2a)
        
        annots = list(self.p2a.values())
        
        self.ted = TextEncoderDecoder(annots, os.path.join(self.path, 'vocab.json'))
        
        self.input_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img < 127] = 0
        img[img >= 127] = 255
        img = img / 255
        return self.input_transform(img) 
        
    def _read_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        return lines 
     
    def _load_annot(self):
        annots = self._read_file(os.path.join(self.path, 'ground_truth.txt'))
        p2a = {}
        
        if self.split == "train":
            print("Loading train annotations...")
            for i, line in tqdm(enumerate(annots)):
                path = f"output_{i}.png"
                p2a[path] = line.strip()
        
        if self.split == "val":
            print("Loading train annotations...")
            for i, line in tqdm(enumerate(annots)):
                path = f"output_{i}.png"
                p2a[path] = line.strip()
            
        return p2a
    
    def __len__(self):
        return self.len_p2a
    
    def __getitem__(self, idx):
        
        path2annot = self.p2a
        
        def get_sample(index):
            path = list(path2annot.keys())[index]
            annot = path2annot[path]
            path = os.path.join(self.path, path)
            img = self._load_image(path)
            return img, annot
        
        img, annot = get_sample(idx)
        tokenized_label = self.ted.encode_text(annot, add_pad=True)
        tokenized_label = tokenized_label.clone().detach()
        
        return img , tokenized_label
    
if __name__ == "__main__":
    # Code to test PretrainingDataset
    # train_dataset = PretrainingDataset("/media/sai/Data/tester/3_imgs_dataset_synthetic/normal_images")
    # print(len(train_dataset))
    # train_dl = DataLoader(train_dataset,batch_size = 4 , shuffle =True)
    # batch = next(iter(train_dl))
    # print(batch.shape)
    # print(len(train_dataset))
    # for images in batch:
    #     pixels = images.permute(1,2,0).numpy()
    #     cv2.imwrite("test.jpg",pixels*255)  # to save the image
    #     break

    # # Code to test PageData
    # root_dir = "/ssd_scratch/chirag_saigunda/"
    # DatasetName = "StackDataset"
    
    # train_datasetstack = PageData("train",f"{root_dir}/{DatasetName}/Train")
    # val_datasetstack = PageData("val",f"{root_dir}/{DatasetName}/Val")
    
    # train_dls = DataLoader(train_datasetstack, batch_size=3, shuffle=True, num_workers=20)
    # valid_dls = DataLoader(val_datasetstack, batch_size=3, shuffle=False,num_workers=20)
    
    # def print_labels(dataloader):
    #     # Get the first batch of images and labels
    #     for batch_idx, batch in enumerate(dataloader):
    #         images, labels = batch
    #         # Save images from the first batch
    #         for i in range(len(images)):
    #             print(train_datasetstack.ted.decode_text(labels[i]))
    #         break  # Remove this line if you want to save all batches

    # print_labels(train_dls)
    # print_labels(valid_dls)
    
    root_dir = "/ssd_scratch/chirag_saigunda/"
    DatasetName = "OverLapStackDataset"
    
    train_datasetoverlap = PageData("train",f"{root_dir}/{DatasetName}/Train")
    val_datasetoverlap = PageData("val",f"{root_dir}/{DatasetName}/Val")
    
    train_dlv = DataLoader(train_datasetoverlap, batch_size=3, shuffle=True, num_workers=20)
    valid_dlv = DataLoader(val_datasetoverlap, batch_size=3, shuffle=False,num_workers=20)
    print(train_datasetoverlap.ted.max_len)
    
    def print_labels(dataloader):
        # Get the first batch of images and labels
        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch
            # Save images from the first batch
            print(images.shape,labels.shape)
            for i in range(len(images)):
                print(train_datasetoverlap.ted.decode_text(labels[i]))
            break  # Remove this line if you want to save all batches

    print_labels(train_dlv)
    print_labels(valid_dlv)

