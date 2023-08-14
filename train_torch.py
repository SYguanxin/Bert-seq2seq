import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import Bert2Bert
from src.trainer import Trainer, TrainerConfig
from src.utils import Noval_Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

########################################################################################################
# Set batch size
########################################################################################################

batch_size = 2

########################################################################################################
# Set learning_rate, training mini-epochs
########################################################################################################

lr_init = 2e-5
lr_final = 1e-6
warmup_tokens = 0

max_epochs = 3
epoch_save_frequency = 1
epoch_save_path = 'src/save_models'

########################################################################################################
# Load data
########################################################################################################

data_root = 'src/data'

train_set = Noval_Dataset(root=data_root, train=False)

########################################################################################################
# set model, tokenizer, data_collator
########################################################################################################

model = Bert2Bert()
tokenizer = train_set.tokenizer

# Set model's config
model.bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
model.bert2bert.config.eos_token_id = tokenizer.eos_token_id
model.bert2bert.config.pad_token_id = tokenizer.pad_token_id

data_collator = DataCollatorForSeq2Seq(tokenizer, model)

########################################################################################################
# train model
########################################################################################################

args = Seq2SeqTrainingArguments(
    output_dir=epoch_save_path,
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=lr_init,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_set,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.evaluate()

trainer.train()