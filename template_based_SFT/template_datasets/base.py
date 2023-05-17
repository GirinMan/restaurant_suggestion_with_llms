from loguru import logger
from torch.utils.data import DataLoader
import torch

class Counter():
    def __init__(self) -> None:
        self.count = 0
        self.total = 0
    
    def add_one(self):
        self.count += 1

    def count_total(self):
        self.total += 1


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {'input_ids': torch.LongTensor(input_ids), 'labels': torch.LongTensor(labels)}

def collate_fn_eval(batch):
    prompt_ids = [item['prompt_ids'] for item in batch]
    prompt_attention_masks = [item['prompt_attention_masks'] for item in batch]
    decoded_labels = [item['decoded_labels'] for item in batch]
    return {'input_ids': torch.LongTensor(prompt_ids), 'attention_mask': torch.LongTensor(prompt_attention_masks), 'decoded_labels': decoded_labels}


class BaseDataset():
    def __init__(self, tokenizer, max_len, batch_size, num_workers, train_split="train", eval_split="validation", test_split="test"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_collator = collate_fn
        self.eval_collator = collate_fn_eval
        self.train_split = train_split
        self.eval_split = eval_split
        self.test_split = test_split
        self.dataset = None
        self.ids_to_labels = None

    def get_train_dataloader(self):
        dataloader = DataLoader(self.dataset[self.train_split],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=True, collate_fn=self.train_collator)
        return dataloader
    
    def get_eval_dataloader(self):
        if self.eval_split is None:
            logger.error("There isn't any eval split in this dataset.")
            return None
        
        dataloader = DataLoader(self.dataset[self.eval_split],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False, collate_fn=self.eval_collator)
        return dataloader
    
    def get_test_dataloader(self):
        if self.test_split is None:
            logger.error("There isn't any test split in this dataset.")
            return None
        
        dataloader = DataLoader(self.dataset[self.test_split],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False, collate_fn=self.eval_collator)
        return dataloader
    