import torch
from torch import nn
from transformers import EncoderDecoderModel, BertModel, BertLMHeadModel, GenerationConfig
import re


class Bert2Bert(nn.Module):
    def __init__(self, tokenizer=None):
        super(Bert2Bert, self).__init__()
        self.bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "src/bert-base-chinese", "src/bert-base-chinese")
        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, num_tokens=None):
        sequence_length = labels.shape[-1]
        index_matrix = torch.arange(sequence_length).view(
            1, sequence_length).to(input_ids.device)
        index_matrix = index_matrix.view(1, 1, sequence_length)
        index_matrix_t = index_matrix.view(1, sequence_length, 1)

        tril = index_matrix <= index_matrix_t
        decoder_attention_mask = (index_matrix < num_tokens.view(-1, 1, 1)) & (
                                            index_matrix_t < num_tokens.view(-1, 1, 1)) & tril

        return self.bert2bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    def generate(self, input_ids, attention_mask):
        config = GenerationConfig(
            max_length=512,
            num_beams=1,
            do_sample=True,
            top_k=0,
            top_p=0.7,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            is_encoder_decoder=True,
        )
        return self.bert2bert.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=config,
        )

    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate,
                                      betas=train_config.betas, eps=train_config.eps)

        return optimizer

    def freeze(self):
        for pn, p in self.named_parameters():
            if not re.search('crossattention', pn):
                p.requires_grad = False

    def in_freeze(self):
        for pn, p in self.named_parameters():
            if not re.search('crossattention', pn):
                p.requires_grad = True


if __name__ == '__main__':
    model = Bert2Bert()
    model.freeze()