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
# Load data
########################################################################################################

data_root = 'src/data'

test_set = Noval_Dataset(root=data_root, train=False)
tokenizer = test_set.tokenizer

model = torch.load('src/save_models/noval-trained-1.pth')
model.tokenizer = tokenizer

# Set model's config
model.bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
model.bert2bert.config.eos_token_id = tokenizer.eos_token_id
model.bert2bert.config.pad_token_id = tokenizer.pad_token_id

########################################################################################################
# Test
########################################################################################################

tconf = TrainerConfig()
trainer = Trainer(model, None, None, config=tconf, tokenizer=tokenizer, train=False)

seq = "“哈哈！真是太巧了！”紫袍青年一上台，就大笑了起来，" \
      "道：“本来以为还要等几场，我们才会遇上，没想到第二场就遇上了！" \
      "”紫袍青年康粱，欣喜无比，不待苏莫说话，急忙转头望向了观战台。“大长老，我与苏莫生死一战，双方完全自愿，请大长老应允！”" \
      "康粱朗声说道，面上带着浓浓的自信。\
      康粱此话一出，全场骤然一静。\
      又是生死之战？\
      天盟要对苏莫发难了么？\
      大长老闻言，眉头大皱。\
      这天盟和苏莫，是没完没了了么？"

trainer.process(title='第一百四十六章 轻松碾压', seq=seq)
for _ in range(10):
      trainer.process()

print(trainer.output)