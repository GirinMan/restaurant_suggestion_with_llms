from loguru import logger
from .base import BaseDataset, Counter
import datasets, re

class SAMSum(BaseDataset):
    def get_prompt_dataset(self, dataset, debug=False, mask_prompt=True):
        tokenizer = self.tokenizer
        prefix = self.prefix
        suffix = self.suffix
        columns = self.columns
        template = self.template
        max_len = self.max_len

        # TODO fix max_label_len to longest summary
        max_label_len=0

        summary = dataset[self.train_split][columns[-1]]
        if self.eval_split:
            summary = summary + dataset[self.eval_split][columns[-1]]
        if self.test_split:
            summary = summary + dataset[self.test_split][columns[-1]]
        tokenized_summary = tokenizer(summary, add_special_tokens=False)['input_ids']

        max_label_len = 0
        for summ in tokenized_summary:
            if len(summ) > max_label_len:
                max_label_len = len(summ)
        if debug:
            logger.info(f"Max label len: {max_label_len}")
        max_label_len = max_label_len + 1
        self.max_label_len = max_label_len

        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'

        tokenized_prefix = tokenizer.encode(prefix, add_special_tokens=False)
        tokenized_suffix = tokenizer.encode(suffix, add_special_tokens=False)
        prefix_len = len(tokenized_prefix)
        suffix_len = len(tokenized_suffix)

        if debug:
            counter = Counter()

        def generate_prompt(examples):
            documents = examples[columns[0]] # dialogue
            labels_origin = examples[columns[1]] # summary

            documents = [template(d) for d in documents]
            labels = [l.strip() + tokenizer.eos_token for l in labels_origin]
            
            tokenized_documents = tokenizer(documents, add_special_tokens=False)['input_ids']
            tokenized_labels = tokenizer(labels, add_special_tokens=False)['input_ids']

            token_ids = []
            label_ids = []
            prompt_ids = []
            prompt_attention_masks = []
            for document_toked, label_toked in zip(tokenized_documents, tokenized_labels):
                document_len = len(document_toked)
                label_len = len(label_toked)
                if debug:
                    counter.count_total()
                if document_len + label_len > max_len:
                    if debug:
                        counter.add_one()
                    label_toked = label_toked[-max_label_len:]
                    document_toked = document_toked[prefix_len:-suffix_len]
                    document_toked = document_toked[-(max_len - prefix_len - suffix_len - max_label_len):]
                    document_toked = tokenized_prefix + document_toked + tokenized_suffix
                    document_len = len(document_toked)
                    label_len = len(label_toked)
                if mask_prompt:
                    label_id = [
                        -100,
                    ] * document_len + label_toked
                else:
                    label_id = document_toked + label_toked
                    for i, id in enumerate(label_id):
                        if id == tokenizer.pad_token_id:
                            label_id[i] = -100
                while len(label_id) < max_len:
                    label_id += [-100]
                token_id = document_toked + label_toked
                while len(token_id) < max_len:
                    token_id += [tokenizer.pad_token_id]
                prompt_attention_mask = [0] * (max_len - document_len) + [1] * document_len
                while len(document_toked) < max_len:
                    document_toked = [tokenizer.pad_token_id] + document_toked

                token_ids.append(token_id)
                label_ids.append(label_id)
                prompt_ids.append(document_toked)
                prompt_attention_masks.append(prompt_attention_mask)

            return {
                'input_ids' : token_ids,
                'labels' : label_ids,
                'prompt_ids' : prompt_ids,
                'prompt_attention_masks' : prompt_attention_masks,
                'decoded_labels' : labels_origin,
            }

        features = dataset["train"].features
        dataset = dataset.map(generate_prompt, batched=True, remove_columns=features)

        if debug:
            logger.info(f"{counter.count} elements has been truncated due to max length limit out of total {counter.total} entities.")
        
        return dataset

    def __init__(self, tokenizer, max_len=64, debug=False, batch_size=4, num_workers=0):
        super().__init__(tokenizer, max_len, batch_size, num_workers, eval_split="validation", test_split="test")
        self.prefix = 'Summarize the following conversation in one sentence.\n'
        self.suffix = '\n<Summary>\n'
        self.template = lambda o: f'{self.prefix}{o.strip()}{self.suffix}'
        self.columns = ['dialogue', 'summary']

        dataset = datasets.load_dataset('samsum')
        self.dataset = self.get_prompt_dataset(dataset=dataset,
                                               debug=debug,
                                               mask_prompt=True)
