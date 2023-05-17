import datetime, os, sys, json, torch, wandb
import evaluate as eval_metrics
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from time import time
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, Trainer, TrainerCallback, set_seed
from loguru import logger
from tqdm import tqdm
from dataset_factory import DatasetFactory

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    train_in_8bit: Optional[bool] = field(
        default=True, metadata={"help": "Train with bitsandbytes or not"}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="nsmc", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_len: Optional[int] = field(
        default=64, metadata={"help": "The maximum length of each data in dataset to use."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        copy_config = True
        json_file = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args = parser.parse_json_file(json_file=json_file)
    else:
        copy_config = False
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    KST = datetime.timezone(datetime.timedelta(hours=9))
    timestamp = str(datetime.datetime.fromtimestamp(time(), tz=KST)).split()

    if not training_args.output_dir.endswith('/'):
        training_args.output_dir = training_args.output_dir + "/"
    training_args.output_dir = training_args.output_dir + timestamp[0] + "_" + timestamp[1][:8]
    training_args.run_name = training_args.run_name + "-" + timestamp[0] + "-" + timestamp[1][:8]

    os.makedirs(training_args.output_dir, exist_ok=True)
    if copy_config:
        with open(json_file, 'r') as openfile:
            config_object = json.load(openfile)
        with open(training_args.output_dir + '/running_config.json', "w") as outfile:
            json.dump(config_object, outfile)
    else:
        configs = sys.argv[2:]
        config_object = {}
        for i in range(int(len(configs)/2)):
            key = configs[i*2].split('--')[-1]
            value = configs[i*2 + 1]
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            config_object[key] = value
        with open(training_args.output_dir + '/running_config.json', "w") as outfile:
            json.dump(config_object, outfile)

    logger.add(training_args.output_dir + '/{time}_train.log')

    logger.info("Model arguments")
    for arg in str(model_args).split(','):
        logger.debug(arg.strip())

    logger.info("Data arguments")
    for arg in str(data_args).split(','):
        logger.debug(arg.strip())

    logger.info("Training arguments")
    for arg in str(training_args).split(','):
        logger.debug(arg.strip())

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.train_in_8bit,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    #model.resize_token_embeddings(len(tokenizer))

    # ### Prepare model for training
    #
    # Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will:
    # - Cast the layer norm in `float32` for stability purposes
    # - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
    # - Enable gradient checkpointing for more memory-efficient training
    # - Cast the output logits in `float32` for smoother sampling during the sampling procedure

    if model_args.train_in_8bit:
        if 'GPTNeoX' in model.config.architectures[0]:
            model = prepare_model_for_int8_training(
                model, output_embedding_layer_name="embed_out", layer_norm_names=["layer_norm", "layernorm"], cast_dtype=torch.float32
            )
        else:
            model = prepare_model_for_int8_training(model, cast_dtype=torch.float32)
    else:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()


    # ### Apply LoRA
    #
    # Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    target_modules = None
    if 'GPTNeoX' in model.config.architectures[0]:
        target_modules = ["query_key_value", "xxx"]  # workaround to use 8bit training on this model
    config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", inference_mode=False
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # load dataset
    df = DatasetFactory(tokenizer)

    loaded_dataset = df.load_dataset(
        name=data_args.dataset_name,
        max_len=data_args.max_len,
        batch_size=training_args.per_device_eval_batch_size
    )

    data = loaded_dataset.dataset
    suffix = loaded_dataset.suffix
    max_label_len = loaded_dataset.max_label_len
  
    if training_args.do_eval:
        rouge = eval_metrics.load('rouge')
        eval_dataloader = loaded_dataset.get_eval_dataloader()
        def validation_step(model, tokenizer, batch, first):
            with torch.cuda.amp.autocast():
                generated_ids = model.generate(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch['attention_mask'].to(model.device),
                    max_new_tokens=max_label_len,
                    eos_token_id = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id,
                )
            generated_txt = []
            for i, g in enumerate(generated_ids):
                decoded_txt = tokenizer.decode(g.tolist(), skip_special_tokens=True)
                generated_txt.append(decoded_txt.strip())
            
            labels = batch['decoded_labels']

            if first:
                count = 0
                for gold, gen_txt in zip(labels, generated_txt):
                    logger.debug(f'gold: {gold} pred: {gen_txt}')
                    count += 1
                    if count > 4:
                        break

            return {'generated': generated_txt, 'labels': labels}

        def validation_epoch_end(outputs):
            generated_txt = []
            labels = []
            preds = []

            for i in outputs:
                generated_txt.extend(i['generated'])
                labels.extend(i['labels'])
            
            metrics = rouge.compute(predictions=generated_txt,
                                    references=labels,
                                    # tokenizer=lambda x: x.split(),
                                    )
            
            return metrics

        def evaluate(eval_dataloader):
            model.eval()
            eval_results = []
            for i, batch in tqdm(enumerate(eval_dataloader), "Generating predictions", total=len(eval_dataloader)):
                eval_results.append(validation_step(model, tokenizer, batch, i == 0))
            model.train()
            return validation_epoch_end(eval_results)

        class EvaluationCallback(TrainerCallback):
            def on_train_begin(self, args: TrainingArguments, state, control, logs=None, **kwargs):
                wandb.log({"rouge1" : 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0, "eval_epoch" : 0.0})
            
            def on_epoch_end(self, args, state, control, logs=None, **kwargs):
                logger.info(f"***Evaluation at epoch {state.epoch} begins***")
                metrics = evaluate(eval_dataloader)
                logger.info(f"***Evaluation results***")
                for key, value in metrics.items():
                    logger.info(f"{key}: {value}")
                metrics['eval_epoch'] = state.epoch
                wandb.log(metrics)
                save_dir = training_args.output_dir + f"/checkpoint-{state.global_step}"
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)

    # ### Training
    if training_args.do_train:
        model.train()
        model.gradient_checkpointing_enable()
        trainer = Trainer(
            model=model,
            train_dataset=data["train"],
            args=training_args,
            data_collator=loaded_dataset.train_collator,
        )
        if training_args.do_eval:
            trainer.add_callback(EvaluationCallback)
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        wandb.finish()
        model.save_pretrained(training_args.output_dir)
        model.config.use_cache = True

    # ## Share adapters on the 🤗 Hub
    # model.push_to_hub(training_args.output_dir, use_auth_token=True)

    # Load adapters from the Hub and generate some output texts:

    # peft_model_id = training_args.output_dir
    # config = PeftConfig.from_pretrained(peft_model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # 
    # # Load the Lora model
    # model = PeftModel.from_pretrained(model, peft_model_id)
    # # You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference

    # batch = tokenizer("다음 제목의 주제를 IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 중 하나로 분류하세요.\n기업들 현금 실탄 쌓자…코로나 위기에 자산 처분 잇따라\n주제:", return_tensors="pt")
    # batch.to('cuda')
    # with torch.cuda.amp.autocast():
    #     output_tokens = model.generate(input_ids = batch['input_ids'], max_new_tokens=data_args.max_label_len + 1)
    # logger.info(f"Generated: {tokenizer.decode(output_tokens[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
