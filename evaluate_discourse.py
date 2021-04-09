from argparse import ArgumentParser
import math

from convokit import Corpus, download
import numpy as np
from scipy.stats import mode
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel, get_linear_schedule_with_warmup

from data import CoarseDiscourseDataset, ConversationPathBatchSampler, add_title_to_root, conversation_path_collate_fn
from model import ConversationClassificationHRNN, UtteranceClassificationHRNN


WARMUP_RATIO = 0.1
CLIPPING_GRADIENT_NORM = 1.0
DISPLAY_STEPS = 100
CORPUS = 'reddit-coarse-discourse-corpus'


parser = ArgumentParser(description='Discourse Evaluation')
parser.add_argument('-m', '--model-name', type=str, default='google/bigbird-roberta-base',
                    help='name of pretrained model to use')
parser.add_argument('-g', '--gpu', type=int, default=None,
                    help='index of gpu to use)')
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
parser.add_argument('--utterance-max', type=int, default=1024,
                    help='maximum utterance length')
parser.add_argument('-p', '--pretrain-path', type=str, default=None,
                    help='path to pretrained model')


def main() -> None:

    args = parser.parse_args()

    if args.gpu is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    corpus = Corpus(filename=download(CORPUS))
    add_title_to_root(corpus)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    conversations = list(corpus.iter_conversations())
    train_ceil = math.ceil(len(conversations) * args.train_split)
    train_conversations = conversations[:train_ceil]
    val_conversations = conversations[train_ceil:]

    train_dataset = CoarseDiscourseDataset(corpus, train_conversations, tokenizer, max_len=args.max_conversation_len, max_tokenization_len=args.utterance_max)
    val_dataset = CoarseDiscourseDataset(corpus, val_conversations, tokenizer, max_len=args.max_conversation_len, max_tokenization_len=args.utterance_max)
    val_dataset.label_encoder = train_dataset.label_encoder
    train_sampler = ConversationPathBatchSampler(args.batch_size, 1, train_dataset.get_indices_by_len())
    val_sampler = ConversationPathBatchSampler(args.batch_size, 1, val_dataset.get_indices_by_len())
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=conversation_path_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=conversation_path_collate_fn, pin_memory=True)

    num_training_steps = args.epochs * len(train_dataset)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(train_dataset.label_encoder.classes_))
    model.to(device)

    if args.pretrain_path is not None:
        checkpoint = torch.load(args.pretrain_path, map_location=device)
        model.bert.load_state_dict(checkpoint['state_dict'])

    optimizer = AdamW(model.parameters(), args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_RATIO * num_training_steps, num_training_steps=num_training_steps)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        print('Epoch {}'.format(epoch))
        train(train_loader, model, optimizer, scheduler, scaler, device, tokenizer.sep_token_id)
        validate(val_loader, model, device, tokenizer.sep_token_id)

def train(loader: DataLoader, model: nn.Module, optimizer: Optimizer, scheduler: object, scaler: GradScaler, device: torch.device, sep_token_id: int):
    non_blocking = device.type != 'cpu'
    loss_fct = nn.CrossEntropyLoss()
    losses = []
    samples_seen = 0
    for i, (path, attention_mask, targets) in enumerate(loader):
        path = path.squeeze(1).to(device, non_blocking=non_blocking)
        attention_mask = path.squeeze(1).to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        with autocast():
            output = model(input_ids=path, attention_mask=attention_mask)
            logits = output.logits[path == sep_token_id]
            loss = loss_fct(logits, targets.reshape(-1))

        losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING_GRADIENT_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        samples_seen += targets.shape[0]
        if (i + 1) % DISPLAY_STEPS == 0 or samples_seen == len(loader.dataset):
            print('Training [{}/{}] Loss: {}'.format(samples_seen, len(loader.dataset), np.mean(losses[-DISPLAY_STEPS:])))

def validate(loader: DataLoader, model: nn.Module, device: torch.device, sep_token_id: int):
    non_blocking = device.type != 'cpu'
    loss_fct = nn.CrossEntropyLoss()
    with torch.no_grad():
        losses = []
        prediction_votes = {}
        for i, (path, attention_mask, targets) in enumerate(loader):
            path = path.squeeze(1).to(device, non_blocking=non_blocking)
            attention_mask = path.squeeze(1).to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            with autocast():
                output = model(input_ids=path, attention_mask=attention_mask)
                logits = output.logits[path == sep_token_id]
                loss = loss_fct(logits, targets.reshape(-1))
                predictions = logits.argmax()

            losses.append(loss.item())

            # baseline
            # predictions = []
            # for id_batch in ids:
            #     predictions.append([])
            #     for id_ in id_batch:
            #         if id_ in loader.dataset.corpus.conversations or '?' in loader.dataset.corpus.get_utterance(id_).text:
            #             prediction = 'question'
            #         else:
            #             prediction = 'answer'
            #         predictions[-1].append(prediction)
            #     predictions[-1] = loader.dataset.label_encoder.transform(predictions[-1])
            #

        #     for id_batch, prediction_batch in zip(ids, predictions):
        #         for utterance_id, prediction in zip(id_batch, prediction_batch):
        #             if utterance_id in prediction_votes:
        #                 prediction_votes[utterance_id].append(prediction)
        #             else:
        #                 prediction_votes[utterance_id] = [prediction]
        # utterance_ids = list(prediction_votes.keys())
        # predictions = np.array([mode(prediction_votes[utterance_id]).mode[0] for utterance_id in utterance_ids])
        # labels = [loader.dataset.corpus.get_utterance(utterance_id).retrieve_meta('majority_type') for utterance_id in utterance_ids]
        # targets = loader.dataset.label_encoder.transform(labels)
        # accuracy = np.mean(targets == predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(targets.to('cpu'), predictions.to('cpu'), average='weighted')
        print('Validation Loss: {}, Acc: {}, Pre: {}, Rec: {}, F: {}'.format(np.mean(losses), accuracy, precision, recall, fscore))


if __name__ == '__main__':
    main()
