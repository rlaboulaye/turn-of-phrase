from argparse import ArgumentParser
import math
from typing import Callable

from convokit import Corpus, download
import numpy as np
from scipy.stats import mode
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModel, AutoTokenizer, PreTrainedModel, get_linear_schedule_with_warmup

from data import ConversationPathBatchSampler, WinningArgumentsDataset, add_title_to_root, conversation_path_collate_fn, filter_winning_arguments_corpus
from model import ConversationClassificationHRNN


WARMUP_RATIO = 0.1
CLIPPING_GRADIENT_NORM = 1.0
DISPLAY_STEPS = 100
CORPUS = 'reddit-coarse-discourse-corpus'


parser = ArgumentParser(description='Discourse Evaluation')
parser.add_argument('-m', '--model-name', type=str, default='albert-base-v2',
                    help='name of pretrained model to use')
parser.add_argument('-g', '--gpu', type=int, default=None,
                    help='index of gpu to use)')
parser.add_argument('--hidden', type=int, default=200,
                    help='hidden size of conversation model')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of layers in conversation model')
parser.add_argument('-l', '--learning-rate', type=float, default=2e-5,
                    help='base learning rate used')
parser.add_argument('-b', '--batch-size', type=int, default=2,
                    help='training data batch size')
parser.add_argument('-e', '--epochs', type=int, default=3,
                    help='number of epochs to run')
parser.add_argument('-t', '--train-split', type=float, default=.8,
                    help='amount of data used for training')
parser.add_argument('--max-conversation-len', type=int, default=11,
                    help='maximum conversation length')
parser.add_argument('--utterance-max', type=int, default=256,
                    help='maximum utterance length')
parser.add_argument('-p', '--pretrain-path', type=str, default=None,
                    help='path to pretrained model')
parser.add_argument('-c', '--corpus', type=str, default='winning-args-corpus',
                    help='evaluation corpus')


def main() -> None:

    args = parser.parse_args()

    if args.gpu is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    corpus = Corpus(filename=download(args.corpus))

    if args.corpus == 'winning-args-corpus':
        corpus = filter_winning_arguments_corpus(corpus)
        DatasetClass = WinningArgumentsDataset
        n_classes = 1
        criterion = nn.BCEWithLogitsLoss()

    add_title_to_root(corpus)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    conversations = list(corpus.iter_conversations())
    train_ceil = math.ceil(len(conversations) * args.train_split)
    train_conversations = conversations[:train_ceil]
    val_conversations = conversations[train_ceil:]

    train_dataset = DatasetClass(corpus, train_conversations, tokenizer, max_len=args.max_conversation_len, max_tokenization_len=args.utterance_max)
    val_dataset = DatasetClass(corpus, val_conversations, tokenizer, max_len=args.max_conversation_len, max_tokenization_len=args.utterance_max)
    train_sampler = ConversationPathBatchSampler(args.batch_size, 1, train_dataset.get_indices_by_len())
    val_sampler = ConversationPathBatchSampler(args.batch_size, 1, val_dataset.get_indices_by_len())
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=conversation_path_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=conversation_path_collate_fn, pin_memory=True)

    num_training_steps = args.epochs * len(train_dataset)

    utterance_encoder = AutoModel.from_pretrained(args.model_name)
    conversation_encoder = nn.LSTM(utterance_encoder.config.hidden_size, 200, args.num_layers)
    pretrained_model = ConversationClassificationHRNN(utterance_encoder, conversation_encoder, 1)
    pretrained_model.to(device)

    if args.pretrain_path is not None:
        checkpoint = torch.load(args.pretrain_path, map_location=device)
        pretrained_model.load_state_dict(checkpoint['state_dict'])

    model = ConversationClassificationHRNN(utterance_encoder, conversation_encoder, n_classes)
    model.hrnn = pretrained_model.hrnn
    del pretrained_model
    model.to(device)

    optimizer = AdamW(model.parameters(), args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_RATIO * num_training_steps, num_training_steps=num_training_steps)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        print('Epoch {}'.format(epoch))
        train(train_loader, model, criterion, optimizer, scheduler, scaler, device)
        validate(val_loader, model, criterion, device)

def train(loader: DataLoader, model: nn.Module, criterion: Callable, optimizer: Optimizer, scheduler: object, scaler: GradScaler, device: torch.device):
    non_blocking = device.type != 'cpu'
    losses = []
    accuracies = []
    samples_seen = 0
    for i, (path, attention_masks, targets, _) in enumerate(loader):
        path = [utterance.to(device, non_blocking=non_blocking) for utterance in path]
        attention_masks = [attention_mask.to(device, non_blocking=non_blocking) for attention_mask in attention_masks]
        targets = targets.to(device, non_blocking=non_blocking)

        with autocast():
            logits = model(path, attention_masks)
            if logits.shape[-1] == 1:
                logits = logits.reshape(-1)
                accuracy = .5 * ((logits.sign() * (targets * 2 - 1)).mean() + 1)
            else:
                # handle multiclass
                pass
            loss = criterion(logits, targets)

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING_GRADIENT_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        samples_seen += targets.shape[0]
        if (i + 1) % DISPLAY_STEPS == 0 or samples_seen == len(loader.dataset):
            print('Training [{}/{}] Loss: {}, Acc: {}'.format(samples_seen, len(loader.dataset), np.mean(losses[-DISPLAY_STEPS:]), np.mean(accuracies[-DISPLAY_STEPS:])))

def validate(loader: DataLoader, model: nn.Module, criterion: Callable, device: torch.device):
    non_blocking = device.type != 'cpu'
    with torch.no_grad():
        losses = []
        accuracies = []
        for i, (path, attention_masks, targets, _) in enumerate(loader):
            path = [utterance.to(device, non_blocking=non_blocking) for utterance in path]
            attention_masks = [attention_mask.to(device, non_blocking=non_blocking) for attention_mask in attention_masks]
            targets = targets.to(device, non_blocking=non_blocking)

            with autocast():
                logits = model(path, attention_masks)
                if logits.shape[-1] == 1:
                    logits = logits.reshape(-1)
                    accuracy = .5 * ((logits.sign() * (targets * 2 - 1)).mean() + 1)
                else:
                    # handle multiclass
                    pass
                loss = criterion(logits, targets)

            losses.append(loss.item())
            accuracies.append(accuracy.item())

        print('Validation Loss: {}, Acc: {}'.format(np.mean(losses), np.mean(accuracies)))


if __name__ == '__main__':
    main()
