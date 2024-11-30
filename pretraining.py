import torch 
from transformers import ViTConfig
import lightning as L
from src.autoencoder import Autoencoder
from torch.utils.data import  DataLoader 
from   utils.dataloader import PretrainingDataset 
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import cv2
import numpy as np
import albumentations as A
import wandb
from albumentations.pytorch import ToTensorV2
import random
import subprocess

# import yaml library
import yaml
import logging
from logging.handlers import RotatingFileHandler

# Setting the random seed for reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_slurm_user(job_id):
    try:
        # Run the squeue command to get the job information
        result = subprocess.run(['squeue', '--job', str(job_id), '--noheader', '--format=%u'], 
                                capture_output=True, text=True, check=True)
        
        # The output will contain the username, we strip any whitespace
        user = result.stdout.strip()
        return user
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while fetching the SLURM job info: {e}")
        return None
    
set_random_seed(42)

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = RotatingFileHandler(log_file, maxBytes=2000, backupCount=10)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

class Model(L.LightningModule):
    def __init__(self, encoder_config,run_name,learning_rate,wandb_logger=None):
        super().__init__()
        self.encoder_config = encoder_config
        self.pageocr = Autoencoder(encoder_config)
        self.run_name = run_name
        self.lr = learning_rate
        self.wandb_logger = wandb_logger

    def weighted_mse_loss(self,output, image, weight_black, weight_white):
        squared_diff = (output - image) ** 2
        weights = torch.ones_like(image).to(image.device) 
        weights[image == 0] = weight_black
        weights[image == 1] = weight_white
        
        weighted_squared_diff = squared_diff * weights
        loss = torch.mean(weighted_squared_diff)
        return loss


    def visualise_results(self, images, outputs, type):
        num_images = images.size(0)

        save_dir = f"./visualisation/{self.run_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        epoch_dir = f"./visualisation/{self.run_name}/epoch_{self.current_epoch}"
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        for i in range(num_images):
            # Original image
            original_img_copy = images[i].clone()
            original_img = original_img_copy.permute(1, 2, 0).detach().cpu().numpy()  # Convert tensor to numpy array
            original_img = (original_img * 255).astype(np.uint8)  # Convert to uint8 for cv2
            cv2.imwrite(f"{epoch_dir}/original_{type}_{i}.jpg", original_img)
            # log the images on wandb
            if self.wandb_logger is not None :
                self.wandb_logger.experiment.log({"original_image": [wandb.Image(original_img, caption=f"{self.current_epoch}_{type}_Original_{i}")]})
            # Output image
            output_img_copy = outputs[i].clone()
            output_img =output_img_copy.permute(1, 2, 0).detach().cpu().numpy()  # Convert tensor to numpy array
            output_img = (output_img * 255).astype(np.uint8)  # Convert to uint8 for cv2
            cv2.imwrite(f"{epoch_dir}/output_{type}_{i}.jpg", output_img)
            # log the images on wandb
            if self.wandb_logger is not None :
                self.wandb_logger.experiment.log({"output_image": [wandb.Image(output_img, caption=f"{self.current_epoch}_{type}_Output_{i}")]})
                
            if i == 5:
                break


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images  = batch # shape batch , channels , H, W
        output = self.pageocr(images)
        weight_black = 3
        weight_white = 1
        loss = self.weighted_mse_loss(output, images , weight_black, weight_white)
        self.log("train_loss",loss)
        if(batch_idx == 0 and self.local_rank == 0 and self.current_epoch % 5 == 0):
            self.visualise_results(images,output,"train")
        return loss

    def validation_step(self,batch,batch_idx):
        images  = batch # shape batch , channels , H, W
        output = self.pageocr(images)
        weight_black = 3
        weight_white = 1
        loss = self.weighted_mse_loss(output, images, weight_black, weight_white)
        self.log("val_loss",loss,sync_dist=True)
        if(batch_idx == 0 and self.local_rank == 0 and self.current_epoch % 5 == 0):
            self.visualise_results(images,output,"valid")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
if __name__ == "__main__":
    
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    root_dir = config['data']['root_dir']
    DatasetName = config['data']['name']
    num_global_tokens =  config['training']['num_global_tokens']
    
    # Height , Width
    img_size = config['data']['img_size']
    IMAGE_SIZE = (img_size[0],img_size[1])
    
    PATCH_SIZE = config['data']['patch_size']
    NUM_CHANNELS = config['training']['num_channels']
    HIDDEN_SIZE = config['training']['hidden_size']
    NUM_HEADS = config['training']['num_heads']
    MAX_EPOCHS = config['training']['max_epochs']
    LR = config['training']['learning_rate']
    
        
    patch_h = config['training']['patch_h']
    patch_w = config['training']['patch_w']
    
    Gnode = os.uname()[1]
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    Ada_account = get_slurm_user(slurm_job_id)
    
    run_name = f"gt_{num_global_tokens}_{patch_h}_{patch_w}_{DatasetName}-A"
    
    encoder_config = ViTConfig(image_size= IMAGE_SIZE, 
                               patch_size = PATCH_SIZE, 
                               num_global_tokens=num_global_tokens, 
                               visualize_encoder_attention_mask=True,
                               hidden_size=HIDDEN_SIZE,
                               num_channels=NUM_CHANNELS,
                               num_attention_heads=NUM_HEADS,
                               patch_h = patch_h,
                               patch_w = patch_w)
    
    wandb_logger = WandbLogger(name=run_name,
                               project="page-ocr-pretraining",
                               log_model=False,
                               save_dir=root_dir,
                               group=DatasetName)
    
    run_link = wandb_logger.experiment.url
    
    # Setup logging
    logger = setup_logging(f"{run_name}_{slurm_job_id}.log")
    logger.info(config)
    logger.info(f"Run link: {run_link}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"SLURM job ID: {slurm_job_id}")
    logger.info(f"Ada account: {Ada_account}")
    
    wandb_logger.log_hyperparams({
        'config': {
        'data': {
            'root_dir': config['data']['root_dir'],
            'name': config['data']['name'],
            'img_size': config['data']['img_size'],
            'patch_size': config['data']['patch_size'],
        },
        'training': {
            'num_global_tokens': config['training']['num_global_tokens'],
            'num_channels': config['training']['num_channels'],
            'hidden_size': config['training']['hidden_size'],
            'num_heads': config['training']['num_heads'],
            'max_epochs': config['training']['max_epochs'],
            'learning_rate': config['training']['learning_rate'],
            'patch_h': config['training']['patch_h'],
            'patch_w': config['training']['patch_w'],
        },
        'system': {
            'Gnode': Gnode,
            'slurm_job_id': slurm_job_id,
            'Ada_account': Ada_account,
        }
    }})
    
    model = Model(encoder_config,run_name,LR,wandb_logger=wandb_logger)
    
    # train model
    transform = A.Compose([
    A.RandomGridShuffle(grid=(2,2), p=0.6),
    ToTensorV2()
    ])
    
    transformval =  A.Compose([
                    ToTensorV2()
            ])
    
    
    train_dataset = PretrainingDataset(f"{root_dir}/{DatasetName}/Train",transform=transform)
    valid_dataset = PretrainingDataset(f"{root_dir}/{DatasetName}/Val",transform=transformval)

    checkpoint_callback = ModelCheckpoint(dirpath=f"{root_dir}/{run_name}", save_top_k=2, monitor="val_loss",save_last=True)
    
    trainer = L.Trainer(devices=-1,
                        strategy='ddp_find_unused_parameters_true',
                        default_root_dir=f"{root_dir}/{run_name}",
                        max_epochs=MAX_EPOCHS,logger=wandb_logger,
                        log_every_n_steps=10,
                        callbacks=[checkpoint_callback] )
    
    # Batch size will change for this 
    train_dl = DataLoader(train_dataset,batch_size = 64, shuffle =True,num_workers=10)
    valid_dl = DataLoader(valid_dataset,batch_size = 64, shuffle =False,num_workers=10)

    trainer.fit(model=model,train_dataloaders=train_dl,val_dataloaders=valid_dl)
