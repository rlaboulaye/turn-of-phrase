from argparse import ArgumentParser
import random
import time
from typing import Callable

from convokit import Corpus, download
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModel, AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

from data import ConversationPathDataset, ConversationPathBatchSampler, conversation_path_collate_fn
from model import ConversationClassificationHRNN
from utils import AverageMeter, ProgressMeter, save_checkpoint


WARMUP_RATIO = 0.1
CLIPPING_GRADIENT_NORM = 1.0
MLM_COEF = 1.0


parser = ArgumentParser(description='Turn of Phrase Pretraining')
parser.add_argument('-m', '--model-name', type=str, default='albert-base-v2',
                    help='name of pretrained model to use')
parser.add_argument('-c', '--corpus', type=str, default='subreddit-changemyview',
                    help='name of convokit corpus used for pretraining')
parser.add_argument('--start-index', type=int, default=None,
                    help='start index for utterance.json')
parser.add_argument('--end-index', type=int, default=None,
                    help='end index for utterance.json')
parser.add_argument('-g', '--gpu', type=int, default=None,
                    help='index of gpu to use)')
parser.add_argument('--hidden', type=int, default=200,
                    help='hidden size of conversation model')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of layers in conversation model')
parser.add_argument('-l', '--learning-rate', type=int, default=2e-5,
                    help='base learning rate used')
parser.add_argument('-b', '--batch-size', type=int, default=2,
                    help='training data batch size')
parser.add_argument('-t', '--training-steps', type=int, default=100000,
                    help='number of training steps to run')
parser.add_argument('--loop-steps', type=int, default=1000,
                    help='number of training steps per train loop')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='random seed for reproducibility')
parser.add_argument('--conversation-min', type=int, default=3,
                    help='minimum conversation length')
parser.add_argument('--conversation-max', type=int, default=6,
                    help='maximum conversation length')
parser.add_argument('--num-neighbors', type=int, default=1,
                    help='the number of contrastive examples used')
parser.add_argument('--utterance-max', type=int, default=256,
                    help='maximum utterance length')
parser.add_argument('-r', '--resume_path', type=str, default=None,
                    help='path to model from which you would like to resume')


best_loss = np.inf


def main() -> None:
    global best_loss
    step = 0

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    if args.start_index is not None or args.end_index is not None:
        start_index = args.start_index
        end_index = args.end_index
        if start_index is None:
            start_index = 0
        if end_index is None:
            corpus = Corpus(filename=download(args.corpus), utterance_start_index=start_index)
        else:
            corpus = Corpus(filename=download(args.corpus), utterance_start_index=start_index, utterance_end_index=end_index)
    else:
        corpus = Corpus(filename=download(args.corpus))

    conversations = list(corpus.iter_conversations())
    # add title to root utterances
    for conversation in conversations:
        utterance = corpus.get_utterance(conversation.id)
        title = conversation.retrieve_meta('title')
        if title is None:
            title = ''
        if utterance.text is None:
            utterance.text = title
        else:
            utterance.text = title + ' ' + utterance.text

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = ConversationPathDataset(corpus, tokenizer,
        min_len=args.conversation_min, max_len=args.conversation_max, n_neighbors=args.num_neighbors, max_tokenization_len=args.utterance_max)
    sampler = ConversationPathBatchSampler(args.batch_size, dataset.min_len, dataset.get_indices_by_len())
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=conversation_path_collate_fn, pin_memory=device.type != 'cpu', num_workers=4)

    utterance_encoder = AutoModel.from_pretrained(args.model_name)
    conversation_encoder = nn.LSTM(utterance_encoder.config.hidden_size, args.hidden, args.num_layers)
    model = ConversationClassificationHRNN(utterance_encoder, conversation_encoder, 1)
    mlm_head = AutoModelForMaskedLM.from_pretrained(args.model_name).predictions
    model.to(device)
    mlm_head.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(list(model.parameters()) + list(mlm_head.parameters()), args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_RATIO*args.training_steps, num_training_steps=args.training_steps)
    scaler = GradScaler()

    if args.resume_path is not None:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path, map_location=device)
            step = checkpoint['step']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            mlm_head.load_state_dict(checkpoint['head_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.resume_path, checkpoint['step']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_path))

    while step < args.training_steps:
        loop_steps = args.loop_steps if args.training_steps - step > args.loop_steps else args.training_steps - step
        loss = train(loader, model, mlm_head, criterion, optimizer, scheduler, scaler,
            device, loop_steps, step // args.loop_steps)
        step += loop_steps

        # checkpoint model every k training loops
        k = 2
        if step % (k * args.loop_steps) == 0 or step == args.training_steps:

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            run_name = '{}.{}.{}.{}.{}'.format(args.model_name, args.corpus, args.conversation_max, args.num_neighbors, args.utterance_max)

            save_checkpoint({
                'step': step,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'head_state_dict': mlm_head.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, run_name)

def train(loader: DataLoader, model: nn.Module, mlm_head: nn.Module, criterion: Callable, optimizer: Optimizer, scheduler: object, scaler: GradScaler, device: torch.device, loop_steps: int, step: int):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    path_losses = AverageMeter('Path Loss', ':.4e')
    mlm_losses = AverageMeter('MLM Loss', ':.4e')
    accuracies = AverageMeter('Accuracy', ':6.2f')
    progress = ProgressMeter(
        loop_steps,
        [batch_time, data_time, losses, path_losses, mlm_losses, accuracies],
        prefix="Epoch: [{}]".format(step))

    model.train()

    end = time.time()
    for i, (path, attention_masks, targets, _) in enumerate(loader):
        data_time.update(time.time() - end)

        non_blocking = device.type != 'cpu'
        path = [utterance.to(device, non_blocking=non_blocking) for utterance in path]
        attention_masks = [attention_mask.to(device, non_blocking=non_blocking) for attention_mask in attention_masks]
        targets = targets.to(device, non_blocking=non_blocking)

        #
        mask_indices = []
        for input_ids_batch, attention_mask_batch in zip(path, attention_masks):
            sequence_lengths_batch = attention_mask_batch.shape[1] - attention_mask_batch.flip(1).argmax(1)
            attention_mask_batch = attention_mask_batch.detach().clone()
            # remove masking for sequence length
            for batch_idx in range(input_ids_batch.shape[0]):
                attention_mask_batch[batch_idx, sequence_lengths_batch[batch_idx]:] = 1
            mask_indices_batch = (attention_mask_batch == 0).nonzero(as_tuple=True)
            mask_indices.append(mask_indices_batch)
        #

        with autocast():
            logits, mask_encodings = model(path, attention_masks, mask_indices)
            logits = logits.reshape(targets.shape[0], -1)
            path_loss = criterion(logits, targets)
            mlm_loss = torch.zeros(1).to(device, non_blocking=non_blocking)
            for mask_encoding_batch, input_ids_batch, mask_indices_batch in zip(mask_encodings, path, mask_indices):
                if len(mask_indices_batch[0]) == 0:
                    continue
                mask_targets = input_ids_batch[mask_indices_batch]
                mask_logits = mlm_head(mask_encoding_batch)
                # gradually take mean
                mlm_loss += criterion(mask_logits, mask_targets) * (1. / len(path))
            loss = path_loss + MLM_COEF * mlm_loss
            accuracy = (logits.argmax(1) == targets).float().mean().item()

        losses.update(loss.item(), targets.shape[0])
        path_losses.update(path_loss.item(), targets.shape[0])
        mlm_losses.update(mlm_loss.item(), targets.shape[0])
        accuracies.update(accuracy, targets.shape[0])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING_GRADIENT_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % (loop_steps // 50) == 0:
            progress.display(i)

        if i == loop_steps - 1:
            break
    return losses.avg


if __name__ == '__main__':
    main()
