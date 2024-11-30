from Levenshtein import distance
import torch
from torchmetrics.text import CharErrorRate as CER, WordErrorRate as WER
import json

def load_vocabulary(load_path):
    with open(load_path, 'r') as file:
        vocabulary_list = json.load(file)
    return vocabulary_list

class TextEncoderDecoder:
    def __init__(self, annots , vocab_path):
        vocab = load_vocabulary(vocab_path)
        vocab.sort()
        self.vocab = vocab
        
        # decoder_start_token_id=0, eos_token_id=2, pad_token_id=1
        self.decoder_start_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 1
        
        self.vocab = ['BOS', 'PAD', 'EOS'] + self.vocab
        self._validate_vocab(annots)
        self.max_len = len(max(annots,key=len)) 
        
        self.vocab_size = len(self.vocab)
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.cer = CER()
        self.wer = WER()
        
    
    def _validate_vocab(self,annots):
        assert 'PAD' in self.vocab, "PAD token missing in vocabulary"
        assert 'BOS' in self.vocab, "BOS token missing in vocabulary"
        assert 'EOS' in self.vocab, "EOS token missing in vocabulary"
        for annot in annots:
            for word in annot:
                assert word in self.vocab, f"{word} not found in vocabulary"
                
    def encode_text(self,text,add_pad=True):
        
        encoding = [self.char2idx[char] for char in text]
        encoding = [0]+ encoding + [2]
        
        if add_pad:
            encoding += [-100] * (self.max_len + 2 - len(encoding))
            
        encoding = torch.tensor(encoding)
        encoding = encoding.long()
        return encoding
    
    def decode_logits(self, logits):
        return logits.argmax(dim=-1)
    
    def decode_batch_logits(self, logits):
        preds = self.decode_logits(logits)
        preds = self.batch_decode_text(preds)
        return preds
    
    def batch_cer_wer(self, preds, targets, is_preds_logits=True):
        preds = self.decode_batch_logits(preds) if is_preds_logits else self.batch_decode_text(preds)
        
        targets = self.batch_decode_text(targets)
        distances = [distance(pred, target) for pred, target in zip(preds, targets)]
        gt_lens = [len(target) for target in targets]

        cer = self.cer(preds, targets)
        wer = self.wer(preds, targets)
        return cer, wer, distances, gt_lens
    
    def decode_text(self, encoding):
        text = []
        
        for idx in encoding:
            if idx not in [0,1,2,-100]:
                try:
                    char = self.idx2char[idx.item()]
                    text.append(char)
                except:
                    text.append("<UNK>")
            elif idx == 2:
                break

        text = ''.join(text)
        return text
    
    def batch_decode_text(self, encodings):
        texts = [self.decode_text(encoding) for encoding in encodings]
        return texts