import torch
import numpy as np 
import random
from transformers import ViTConfig, TrOCRConfig
from utils.dataloader import PageData
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from src.pageocr import PageOCR
from torch.nn import functional as F
import subprocess
import os

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
    def __init__(self, encoder_config, decoder_config, ted,error_rate ,saved_check=None,custom_lr=None,freeze_encoder=False):
        super().__init__()
        self.model = PageOCR(encoder_config, decoder_config)
        self.ted = ted
        self.lr = custom_lr
        self.freeze_encoder = freeze_encoder
        self.error_rate = error_rate
        if saved_check is not None:
            checkpoint = torch.load(saved_check, map_location=torch.device('cpu'))  # Load checkpoint to CPU by default
            state_dict = checkpoint['state_dict']
            encoder_state_dict = {k[16:]: v for k, v in state_dict.items() if k.startswith("pageocr.encoder")}
            self.model.pageocr.encoder.load_state_dict(encoder_state_dict, strict=False)

            if self.freeze_encoder:
                for param in self.model.pageocr.encoder.parameters():
                    param.requires_grad = False

            del encoder_state_dict
            del state_dict
            del checkpoint
            print("Loaded model from checkpoint")
            
    def apply_teacher_forcing(self, gt_labels, error_rate):
        gt_labels_error = gt_labels.clone()
        gt_labels_for_mask = gt_labels.clone()
        # get them to cpu
        gt_labels_for_mask = gt_labels_for_mask.cpu()
        valid_mask = (gt_labels_for_mask != -100) & (gt_labels_for_mask != self.ted.decoder_start_token_id) & (gt_labels_for_mask != self.ted.eos_token_id) & (
                        torch.arange(gt_labels_for_mask.size(1)).unsqueeze(0).expand_as(gt_labels_for_mask) < gt_labels_for_mask.size(0)
                    )
                    
        # Create a mask of random values to determine where to apply errors
        random_mask = (torch.rand_like(gt_labels_for_mask.float()) < error_rate) & valid_mask
        
        # Apply random errors to the valid positions
        random_tokens = torch.randint(3, self.ted.vocab_size, gt_labels_for_mask.size(), dtype=gt_labels.dtype)
        # send random tokens to the same device as gt_labels
        random_tokens = random_tokens.to(gt_labels.device)
        gt_labels_error[random_mask] = random_tokens[random_mask]
        
        return gt_labels_error     
            
    def training_step(self, batch, batch_idx):
        
        pixel_values, gt_labels = batch
        if self.error_rate > 0:          
            gt_labels_error = self.apply_teacher_forcing(gt_labels, self.error_rate)        
            output = self.model(pixel_values, gt_labels_error)
            logits = output.logits
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits.reshape(-1, self.ted.vocab_size), gt_labels.reshape(-1))
        else:
            output = self.model(pixel_values, gt_labels)
            logits = output.logits
            loss = output.loss
            
        
        # Compute CER and WER
        cer, wer, _, _ = self.ted.batch_cer_wer(logits, gt_labels)
        
        # Log metrics
        self.log("train_loss", loss,prog_bar=True)
        self.log("train_cer", cer,prog_bar=True)
        self.log("train_wer", wer,prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values, gt_labels = batch
        output = self.model.pageocr.generate(pixel_values, 
                                             decoder_start_token_id=self.ted.decoder_start_token_id,
                                             eos_token_id = self.ted.eos_token_id, 
                                             pad_token_id = self.ted.pad_token_id, 
                                             max_length=self.ted.max_len + 2, 
                                             num_beams=4, early_stopping=True, 
                                             no_repeat_ngram_size=3, length_penalty=2.0)

        cer, wer, _, _ = self.ted.batch_cer_wer(output, gt_labels, is_preds_logits=False)
        
        self.log("val_cer", cer,sync_dist=True,prog_bar=True)
        self.log("val_wer", wer,sync_dist=True,prog_bar=True)

        return cer
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    
if __name__ == "__main__":

    config_path = 'config.yaml'
    config = load_config(config_path)
    
    root_dir = config['data']['root_dir']
    DatasetName = config['data']['name']
    num_global_tokens =  config['training']['num_global_tokens']
    
    # Height , Width
    IMAGE_SIZE = config['data']['img_size']
    
    PATCH_SIZE = config['data']['patch_size']
    NUM_CHANNELS = config['training']['num_channels']
    HIDDEN_SIZE = config['training']['hidden_size']
    NUM_HEADS = config['training']['num_heads']
    MAX_EPOCHS = config['training']['max_epochs']
    LR = config['training']['learning_rate']
    error_rate = config['training']['error_rate']
    
        
    patch_h = config['training']['patch_h']
    patch_w = config['training']['patch_w']
    
    run_name = f"gt_{num_global_tokens}_{patch_h}_{patch_w}_{DatasetName}-ocr-{error_rate}"
    
    train_dataset = PageData('train', path=f'{root_dir}/{DatasetName}/Train')
    val_dataset = PageData('val', path=f'{root_dir}/{DatasetName}/Val')
    textencoderdecoder = train_dataset.ted
            
    encoder_config = ViTConfig(image_size= IMAGE_SIZE, 
                               patch_size = PATCH_SIZE, 
                               num_global_tokens=num_global_tokens, 
                               visualize_encoder_attention_mask=True,
                               hidden_size=HIDDEN_SIZE,
                               num_channels=NUM_CHANNELS,
                               num_attention_heads=NUM_HEADS,
                               patch_h = patch_h,
                               patch_w = patch_w)
    
    decoder_config = TrOCRConfig(
        vocab_size = train_dataset.ted.vocab_size,
        d_model = 512,
        max_position_embeddings = textencoderdecoder.max_len+2,
        num_global_tokens=num_global_tokens,
        decoder_start_token_id = textencoderdecoder.decoder_start_token_id,
        pad_token_id = textencoderdecoder.pad_token_id,
        eos_token_id = textencoderdecoder.eos_token_id,
        bos_token_id = textencoderdecoder.decoder_start_token_id,
    )
    

    Gnode = os.uname()[1]
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    Ada_account = get_slurm_user(slurm_job_id)
    
    notes=f"{num_global_tokens}_gt on {DatasetName},with {error_rate}"
    
    
    wandb_logger = WandbLogger(name=run_name,
                               project="page-ocr",
                               log_model=False,
                               notes=notes,
                               group=f"{DatasetName}")
    
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
            'error_rate': config['training']['error_rate'],
            'saved_check': config['training']['saved_check']
        },
        'system': {
            'Gnode': os.uname()[1],
            'slurm_job_id': os.getenv('SLURM_JOB_ID'),
            'Ada_account': get_slurm_user(os.getenv('SLURM_JOB_ID')),
        }
    }})
    
    saved_check = config['training']['saved_check']
    if saved_check is None:
        raise ValueError("Please provide a saved checkpoint to start training")

    model = Model(encoder_config,
                  decoder_config,
                  ted = textencoderdecoder,
                  error_rate=error_rate,
                  saved_check=saved_check,
                  custom_lr=LR)
    # Stack batch size is 40 , Line it is 80
    
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=20)
    valid_dl = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=20)
    
    checkpoint_val_callback = ModelCheckpoint(dirpath=f"{root_dir}/{run_name}", save_top_k=2, monitor="val_cer", mode="min",filename="ocr-{epoch}-{val_cer:.2f}",save_last=True)
    checkpoint_train_callback = ModelCheckpoint(dirpath=f"{root_dir}/{run_name}", save_top_k=2, monitor="train_cer", mode="min",filename="ocr-{epoch}-{train_cer:.2f}")
    checkpoint_train_loss_callback = ModelCheckpoint(dirpath=f"{root_dir}/{run_name}", save_top_k=2, monitor="train_loss", mode="min",filename="ocr-{epoch}-{train_loss:.2f}")
    
    trainer = L.Trainer(devices=-1,strategy=DDPStrategy(find_unused_parameters=True),
                        default_root_dir=f"{root_dir}/{run_name}",
                        max_epochs=MAX_EPOCHS,log_every_n_steps=10,
                        callbacks=[checkpoint_val_callback,checkpoint_train_callback,checkpoint_train_loss_callback],
                        logger=wandb_logger,
                        num_sanity_val_steps=2)
    
    trainer.fit(model=model,train_dataloaders=train_dl, val_dataloaders=valid_dl)
