import torch 
from transformers import ViTConfig
import lightning as L
from src.autoencoder import Autoencoder
import torch.nn.functional as F
from torch.utils.data import  DataLoader 
from   utils.dataloader import PretrainingDataset 
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import torchvision
import os
import cv2
import numpy as np

class Model(L.LightningModule):
    def __init__(self, encoder_config,run_name,custon_lr):
        super().__init__()
        self.encoder_config = encoder_config
        self.pageocr = Autoencoder(encoder_config)
        self.run_name = run_name
        self.lr = custon_lr

    def weighted_mse_loss(self,output, image, weight_black, weight_white):
        squared_diff = (output - image) ** 2
        weights = torch.ones_like(image).to(image.device) 
        weights[image == 0] = weight_black
        weights[image == 1] = weight_white
        
        weighted_squared_diff = squared_diff * weights
        loss = torch.mean(weighted_squared_diff)
        return loss

    def visualise_results(self,images, outputs,type):
        num_images = images.size(0)
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 4))

        for i in range(num_images):
            # Original image
            original_img = torchvision.transforms.functional.to_pil_image(images[i])
            axes[i, 0].imshow(original_img, cmap='gray')
            axes[i, 0].set_title('Original')

            # Output image
            output_img = torchvision.transforms.functional.to_pil_image(outputs[i])
            axes[i, 1].imshow(output_img, cmap='gray')
            axes[i, 1].set_title('Output')

        plt.tight_layout()
        save_dir = f"/home2/chiragp/saigunda/page-ocr/visualisation/{self.run_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the plot
        plt.savefig(f"{save_dir}/{type}_{self.current_epoch}")
        plt.close()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images  = batch # shape batch , channels , H, W
        output = self.pageocr(images)
        weight_black = 10
        weight_white = 1
        loss = self.weighted_mse_loss(output, images , weight_black, weight_white)
        self.log("train_loss",loss)
        if(batch_idx == 0 and self.local_rank == 0):
            self.visualise_results(images,output,"train")
        return loss

    def validation_step(self,batch,batch_idx):
        images  = batch # shape batch , channels , H, W
        output = self.pageocr(images)
        weight_black = 10
        weight_white = 1
        loss = self.weighted_mse_loss(output, images, weight_black, weight_white)
        self.log("val_loss",loss)
        if(batch_idx == 0 and self.local_rank == 0):
            self.visualise_results(images,output,"valid")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    vocab_size = 64
    num_global_tokens = 4
    LR = 1e-5
    root_dir = "/ssd_scratch/cvit/saigunda/"
    # Height , Width
    IMAGE_SIZE = (32,512)
    PATCH_SIZE = 16
    NUM_CHANNELS = 1
    HIDDEN_SIZE = 768
    NUM_HEADS = 12
    MAX_EPOCHS = 100
    patch_h = 2
    patch_w = 8
    encoder_config = ViTConfig(image_size= IMAGE_SIZE, 
                            patch_size = PATCH_SIZE, 
                            num_global_tokens=num_global_tokens, 
                            visualize_encoder_attention_mask=True,
                            hidden_size=HIDDEN_SIZE,
                            num_channels=NUM_CHANNELS,
                            num_attention_heads=NUM_HEADS,
                            patch_h = patch_h,
                            patch_w = patch_w)
    saved_check = "/ssd_scratch/cvit/saigunda/lr_1e-05_gt_4_2_8_LineDataset/epoch=343-step=127968.ckpt"
    checkpoint = torch.load(saved_check,map_location="cpu")
    # get the keys of the checkpoint
    print(checkpoint.keys())
    # load model 
    model = Model(encoder_config,"test",LR)
    model.load_state_dict(checkpoint['state_dict'])
    tester_dataset = PretrainingDataset(f"{root_dir}/LineDataset/Val")

    tester_dl = DataLoader(tester_dataset,batch_size = 5 , shuffle=False,num_workers=10)
    model.eval()
    with torch.no_grad():
        for batch in tester_dl:
            images = batch
            output = model.pageocr(images)

            for i in range(images.size(0)):

                original_img = torchvision.transforms.functional.to_pil_image(images[i])
                original_img.save(f"results/original_{i}.png")
                
                # torch utils
                torchvision.utils.save_image(output[i].squeeze(),f"results/output_{i}.png")
            break

