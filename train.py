import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.Models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokeinizers/tokeinizer_{0}.json'
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # keet 90 for training 10 for validation
    train_ds_size = int(0.9 * len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])
