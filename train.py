import torchvision
import torch
from torch.utils import data
from torchvision import transforms
from src.model import Bert2Bert
from src.trainer import Trainer, TrainerConfig
from src.utils import Noval_Dataset
from transformers import DataCollatorForSeq2Seq
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

max_epochs = 2
epoch_save_frequency = 1
epoch_save_path = 'src/save_models/noval-trained-'

########################################################################################################
# Load data
########################################################################################################

data_root = 'src/data'

train_set = Noval_Dataset(root=data_root, train=True, long_target=False)

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

train_set.DataCollator = data_collator

########################################################################################################
# train model
########################################################################################################

tconf = TrainerConfig(max_epochs=max_epochs, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path,
                      batch_size=batch_size, betas=(0.9, 0.99), eps=1e-8, grad_norm_clip=1.0, weight_decay=0.01,
                      learning_rate=lr_init, lr_decay=True, lr_final=lr_final,
                      warmup_tokens=warmup_tokens, final_tokens=max_epochs*len(train_set),
                      collate_fn=train_set.collate)
trainer = Trainer(model, train_set, None, tconf)
trainer.train()