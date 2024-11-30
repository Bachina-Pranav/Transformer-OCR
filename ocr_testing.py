import torch
from transformers import ViTConfig, TrOCRConfig
from utils.data_loader import PageData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.ted import TextEncoderDecoder
from src.pageocr import PageOCR
import torch.nn as nn
# use tqdm for progress bar
from tqdm import tqdm
import lightning as L
from torch.utils.data import Subset


class Model(L.LightningModule):
    def __init__(self, encoder_config, decoder_config, ted, saved_check=None,custom_lr=None,freeze_encoder=False):
        super().__init__()
        self.model = PageOCR(encoder_config, decoder_config)
        self.ted = ted
        self.lr = custom_lr
        self.freeze_encoder = freeze_encoder

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
            
    def training_step(self, batch, batch_idx):
        
        pixel_values, gt_labels = batch
        output = self.model(pixel_values, gt_labels)
        loss = output.loss
        logits = output.logits
        
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


def evaluate_model(model, data_loader, device, ted):
    model.to(device)
    total_cer = 0
    first_batch_results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            pixel_values, gt_labels = batch
            pixel_values = pixel_values.to(device)
            gt_labels = gt_labels.to(device)

            output = model.model.pageocr.generate(pixel_values, 
                                             decoder_start_token_id=ted.decoder_start_token_id,
                                             eos_token_id = ted.eos_token_id, 
                                             pad_token_id = ted.pad_token_id, 
                                             max_length=ted.max_len + 2, 
                                             num_beams=4, early_stopping=True, 
                                             no_repeat_ngram_size=3, length_penalty=2.0)


            cer, wer, _, _ = ted.batch_cer_wer(output, gt_labels, is_preds_logits=False)
            print(f"CER: {cer}")
            total_cer += cer

            if i == 0:  # Save results of the first batch
                decoded_preds = ted.batch_decode_text(output)
                decoded_gt_labels = ted.batch_decode_text(gt_labels)
                first_batch_results = list(zip(decoded_preds, decoded_gt_labels))
            break

    # Print first batch results
    for pred, gt in first_batch_results:
        print(f"Predicted: {pred}\nGround Truth: {gt}\n")
           
if __name__ == "__main__":
    root_dir = "/ssd_scratch/chirag_saigunda/"
    DatasetName = "StackDataset"
    #DatasetName = "OverLapStackDataset"
    num_global_tokens = 8
    # Height , Width
    IMAGE_SIZE = (256,128)
    PATCH_SIZE = 16
    NUM_CHANNELS = 1
    HIDDEN_SIZE = 768
    NUM_HEADS = 12
    MAX_EPOCHS = 10000
    LR = 1e-5

    train_dataset = PageData('train', path=f'{root_dir}/{DatasetName}/Train')
    val_dataset = PageData('val', path=f'{root_dir}/{DatasetName}/Val')
    textencoderdecoder = train_dataset.ted
    
    encoder_config = ViTConfig(image_size= IMAGE_SIZE,
                               patch_size = PATCH_SIZE,
                               num_global_tokens=num_global_tokens, 
                               visualize_encoder_attention_mask=False,
                               hidden_size=HIDDEN_SIZE,
                               num_channels=NUM_CHANNELS,
                               num_attention_heads=NUM_HEADS,
                               return_dict=True)
    
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
    

    model = Model(encoder_config, decoder_config, ted=textencoderdecoder)
    
    checkpoint = torch.load(f"ocr-epoch=0-val_cer=0.03.ckpt", map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(train_dataset, batch_size=156, shuffle=True, num_workers=20)
    valid_dl = DataLoader(val_dataset, batch_size=156, shuffle=False,num_workers=20)
    evaluate_model(model, valid_dl, device,ted=textencoderdecoder)
