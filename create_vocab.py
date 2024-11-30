import json

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def compare_files(file1_path, file2_path):
    content1 = read_file(file1_path)
    content2 = read_file(file2_path)
    return content1 == content2

def create_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    unique_chars = set(text)
    unique_chars.discard('\n')
    unique_chars.discard('\r')

    sorted_vocab = sorted(unique_chars)
    return sorted_vocab

def compare_vocab(file1_path, file2_path):
    vocab1 = create_vocab(file1_path)
    vocab2 = create_vocab(file2_path)
    return vocab1 == vocab2

def create_and_save_vocab(file_path, save_path):
    vocab = create_vocab(file_path)
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(vocab, file, ensure_ascii=False, indent=4)

def compare_vocab_files(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as file:
        vocab1 = json.load(file)
    with open(file2_path, 'r', encoding='utf-8') as file:
        vocab2 = json.load(file)
    return vocab1 == vocab2

root_dir = "/ssd_scratch/cvit/kesav_saigunda/"
DatasetName = "2WordsDataset"
Train = f"/{root_dir}/{DatasetName}/Train/ground_truth.txt"
TrainVocab = f"/{root_dir}/{DatasetName}/Train/vocab.json"
Val = f"/{root_dir}/{DatasetName}/Val/ground_truth.txt"
ValVocab = f"/{root_dir}/{DatasetName}/Val/vocab.json"



create_and_save_vocab(Train, TrainVocab)
create_and_save_vocab(Val,ValVocab)

# Compare all vocab files
vocab_comparison_1 = compare_vocab_files(TrainVocab, ValVocab)

print(f"Vocabularies of Stack_Train and Stack_Val are {'the same' if vocab_comparison_1 else 'different'}.")

train_val_comparision_result = compare_vocab(TrainVocab,ValVocab)
print(f"Vocabularies of TrainVocab and ValVocab are {'the same' if train_val_comparision_result else 'different'}.")