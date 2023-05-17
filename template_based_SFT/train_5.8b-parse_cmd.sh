#!/bin/bash
export WANDB_LOG_MODEL="false"
export WANDB_PROJECT="retaurant_suggestion_with_llms"
export WANDB_WATCH="false"
python /home/girinman/repos/restaurant_suggestion_with_llms/template_based_SFT/finetune_peft_generation.py \
/home/girinman/repos/restaurant_suggestion_with_llms/template_based_SFT/configs/polyglot-ko-5.8b_parse_cmd.json
