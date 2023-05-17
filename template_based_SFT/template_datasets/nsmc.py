from loguru import logger
from .base import BaseDataset, Counter
import datasets

class NSMC(BaseDataset):
    def get_prompt_dataset(self, dataset, debug=False):
        tokenizer = self.tokenizer
        prefix = self.prefix
        suffix = self.suffix
        columns = self.columns
        template = self.template
        max_len = self.max_len
        ids_to_labels = self.ids_to_labels

        labels = [value for _, value in ids_to_labels.items()]
        tokenized_labels = tokenizer(labels, add_special_tokens=False)['input_ids']

        max_label_len=0
        for label in tokenized_labels:
            if len(label) > max_label_len:
                max_label_len = len(label)
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
            documents = examples[columns[0]]
            documents = [template(d) for d in documents]
            labels = examples[columns[1]]
            if ids_to_labels is not None:
                labels = [ids_to_labels[i] for i in labels]
            labels = [l + tokenizer.eos_token for l in labels]
            
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
                    document_toked = document_toked[:max_len - prefix_len - suffix_len - max_label_len]
                    document_toked = tokenized_prefix + document_toked + tokenized_suffix
                    document_len = len(document_toked)
                    label_len = len(label_toked)
                label_id = [
                    -100,
                ] * document_len + label_toked
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
                'decoded_labels' : examples[columns[1]],
            }

        features = dataset["train"].features
        dataset = dataset.map(generate_prompt, batched=True, remove_columns=features)

        if debug:
            logger.info(f"{counter.count} elements has been truncated due to max length limit out of total {counter.total} entities.")
        
        return dataset

    def __init__(self, tokenizer, max_len=64, debug=False, batch_size=4, num_workers=0):
        super().__init__(tokenizer, max_len, batch_size, num_workers, eval_split="test", test_split=None)
        self.ids_to_labels = {0 : "부정", 1 : "긍정"}
        self.prefix = '다음 문장은 긍정일까요 부정일까요?\n'
        self.suffix = '\n정답:'
        self.template = lambda o: f'{self.prefix}{o.strip()}{self.suffix}'
        self.columns = ['document', 'label']

        dataset = datasets.load_dataset('nsmc')
        self.dataset = self.get_prompt_dataset(dataset=dataset,
                                               debug=debug)
