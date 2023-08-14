import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import *
import math
import inspect
from torch.cuda.amp import GradScaler, autocast


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    weight_decay = 0.01
    lr_decay = False  # linear warmup followed by cosine decay
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper
    final_tokens = 260e9  # at which point do we reach lr_final
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, tokenizer=None, train=True):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.steps = 0
        self.tokenizer = tokenizer
        self.seq = None
        self.title = ''
        self.output = ''

        self.device = 'cpu'
        if torch.cuda.is_available():  # take over whatever gpus are on the system
            self.device = torch.cuda.current_device()
            if train:
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

    def get_run_name(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        cfg = raw_model.config
        run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + cfg.model_type + '-' + str(
            cfg.n_layer) + '-' + str(cfg.n_embd)
        return run_name

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            if isinstance(data, DataLoader):
                loader = data
            else:
                loader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    collate_fn=config.collate_fn)

            pbar = tqdm(enumerate(loader), total=len(loader),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)

            for it, batch in pbar:
                with torch.set_grad_enabled(is_train):
                    with autocast():
                        output = model(**batch)  # forward the model
                        loss = output.loss

                if is_train:  # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    if config.lr_decay:  # decay the learning rate based on our progress
                        # number of tokens processed this step (i.e. label is not -100)
                        self.tokens += config.batch_size
                        lr_final_factor = config.lr_final / config.learning_rate
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = lr_final_factor + \
                                (1 - lr_final_factor) * float(self.tokens) / \
                                float(config.warmup_tokens)
                            progress = 0
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor /
                                                                     2) * math.cos(math.pi * progress)  # better 1.0 ~ 0.1
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    now_loss = loss.item()  # report progress
                    self.steps += 1

                    if self.avg_loss < 0:
                        self.avg_loss = now_loss
                    else:
                        # factor = max(1.0 / 300, 1.0 / math.sqrt(it + 1))
                        factor = 1 / (it + 1)
                        self.avg_loss = self.avg_loss * (1.0 - factor) + now_loss * factor
                    pbar.set_description(
                        f"mini-epoch {epoch + 1} "
                        f"prog {progress:.2%} iter {it}: ppl {math.exp(self.avg_loss):.2f} "
                        f"loss {self.avg_loss:.4f} lr {lr:e}")

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')

            if (self.config.epoch_save_frequency > 0 and (epoch + 1) % self.config.epoch_save_frequency) == 0 or (
                    epoch == config.max_epochs - 1):
                raw_model = self.model.module if hasattr(self.model,
                                                         "module") else self.model  # DataParallel wrappers keep raw model object in .module
                torch.save(raw_model, self.config.epoch_save_path + str(epoch + 1) + '.pth')

    def test(self):
        is_train = False

        model, config = self.model, self.config
        model.eval()

        data = self.test_dataset
        if isinstance(data, DataLoader):
            loader = data
        else:
            loader = DataLoader(data, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                collate_fn=config.collate_fn)

        for batch in loader:

            labels = batch['labels']
            labels = torch.where(labels >= 0, labels, self.tokenizer.pad_token_id)
            labels = self.tokenizer.batch_decode(labels)

            with torch.set_grad_enabled(is_train):
                keys = inspect.signature(model.generate).parameters
                output = model.generate(**{k: v.to(self.device) for k, v in batch.items() if k in keys})

            for orin, label, logit in zip(
                    self.tokenizer.batch_decode(batch['input_ids']),
                    labels,
                    self.tokenizer.batch_decode(output)):
                print("原句: ", orin)
                print("续写: ", logit)
                print("原续写: ", label)

    def process(self, seq=None, title=None):
        if seq:
            self.seq = seq
            self.output = ''
        elif self.seq is None:
            raise "未输入过句子，请先将句子作为参数传入"
        if title:
            self.title = title

        is_train = False

        model, config = self.model, self.config
        model.eval()

        if not title:
            data = self.tokenizer(self.seq, max_length=512, truncation=True, return_tensors='pt')
        else:
            data = self.tokenizer(self.title + '[SEP]' + self.seq, max_length=512, truncation=True, return_tensors='pt')

        with torch.set_grad_enabled(is_train):
            keys = inspect.signature(model.generate).parameters
            output = model.generate(**{k: v.to(self.device) for k, v in data.items() if k in keys})

        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output = output.replace(' ', '')
        self.output += output
        self.seq += output
        if len(self.title) + len(self.seq) > 509:
            self.seq = self.seq[len(self.title) - 509:]