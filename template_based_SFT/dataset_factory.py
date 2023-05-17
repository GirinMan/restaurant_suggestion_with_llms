from loguru import logger
from template_datasets.nsmc import NSMC
from template_datasets.mrpc import MRPC
from template_datasets.rte import RTE
from template_datasets.mnli import MNLI
from template_datasets.boolq import BoolQ
from template_datasets.hellaswag import HellaSWAG
from template_datasets.samsum import SAMSum
from template_datasets.parse_cmd import CMD

class DatasetFactory():
    def __init__(self, tokenizer):
        self.datasets = {
            "nsmc" : NSMC,
            "mrpc" : MRPC,
            "rte" : RTE,
            "mnli" : MNLI,
            "boolq" : BoolQ,
            "hellaswag" : HellaSWAG,
            "samsum" : SAMSum,
            "parse_cmd" : CMD,
        }
        self.tokenizer = tokenizer

    def show_datasets(self):
        logger.info("=== Available dataset list ===")
        for i, key in enumerate(self.datasets):
            logger.info(f"{i}: {key}")

    def load_dataset(self, name, max_len, batch_size, debug=False):
        if name not in self.datasets:
            logger.error(f"Requested dataset {name} is unavailable.")
            return None
        else:
            return self.datasets[name](tokenizer=self.tokenizer,
                                        max_len=max_len,
                                        batch_size=batch_size,
                                        debug=debug
                                        )


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")

    df = DatasetFactory(tokenizer)
    
    df.show_datasets()
    
    loaded = df.load_dataset('imdb', max_len=160, batch_size=2)
    loaded = df.load_dataset('parse_cmd', max_len=96, batch_size=1, debug=True)

    dataset = loaded.dataset
    train = loaded.get_train_dataloader()
    eval = loaded.get_eval_dataloader()
    test = loaded.get_test_dataloader()

    for batch in train:
        print(batch)
        decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        for text in decoded_inputs:
            logger.info(text)
        break

    for batch in eval:
        print(batch)
        decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        for text in decoded_inputs:
            logger.info(text)
        break
    
    if test is not None:
        for batch in test:
            decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
            for text in decoded_inputs:
                logger.info(text)
            break
    
    print("End")
