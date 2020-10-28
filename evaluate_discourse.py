import math

from convokit import Corpus, download
import numpy as np
from scipy.stats import mode
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModel, AutoTokenizer, PreTrainedModel, get_linear_schedule_with_warmup

from data import CoarseDiscourseDataset, ConversationPathBatchSampler, conversation_path_collate_fn
from model import ConversationClassificationHRNN, UtteranceClassificationHRNN


WARMUP_RATIO = 0.1
CLIPPING_GRADIENT_NORM = 1.0
learning_rate = 1e-4
max_len = 11
batch_size = 2
max_tokenization_len = 256
train_split = .95
epochs = 5

model_name = 'albert-base-v2'
device = torch.device('cuda:0')
non_blocking = device.type != 'cpu'

resume_path = None
# resume_path = 'checkpoints/albert-base-v2.subreddit-changemyview.8.3.256.million_samples.new.checkpoint.pth.tar'

corpus = Corpus(filename=download('reddit-coarse-discourse-corpus'))
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

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ceil = math.ceil(len(conversations) * train_split)
train_conversations = conversations[:train_ceil]
val_conversations = conversations[train_ceil:]

train_dataset = CoarseDiscourseDataset(corpus, train_conversations, tokenizer, max_len=max_len, max_tokenization_len=max_tokenization_len)
val_dataset = CoarseDiscourseDataset(corpus, val_conversations, tokenizer, max_len=max_len, max_tokenization_len=max_tokenization_len)
train_sampler = ConversationPathBatchSampler(batch_size, 1, train_dataset.get_indices_by_len())
val_sampler = ConversationPathBatchSampler(batch_size, 1, val_dataset.get_indices_by_len())
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=conversation_path_collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=conversation_path_collate_fn, pin_memory=True)

num_training_steps = epochs * len(train_dataset)

utterance_encoder = AutoModel.from_pretrained(model_name)
conversation_encoder = nn.LSTM(utterance_encoder.config.hidden_size, 200)
pretrained_model = ConversationClassificationHRNN(utterance_encoder, conversation_encoder, 1)
pretrained_model.to(device)

if resume_path is not None:
    checkpoint = torch.load(resume_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint['state_dict'])

model = UtteranceClassificationHRNN(utterance_encoder, conversation_encoder, len(train_dataset.label_encoder.classes_))
model.hrnn = pretrained_model.hrnn
del pretrained_model
model.to(device)

optimizer = AdamW(model.parameters(), learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_RATIO * num_training_steps, num_training_steps=num_training_steps)
scaler = GradScaler()


for epoch in range(epochs):
    losses = []
    for i, (path, attention_masks, targets, ids) in enumerate(train_loader):
        path = [utterance.to(device, non_blocking=non_blocking) for utterance in path]
        attention_masks = [attention_mask.to(device, non_blocking=non_blocking) for attention_mask in attention_masks]
        targets = targets.to(device, non_blocking=non_blocking)

        with autocast():
            loss = -model(path, attention_masks, targets)

        losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING_GRADIENT_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if (i + 1) % 100 == 0:
            print('Epoch {} Train Loss'.format(epoch))
            print(np.mean(losses))
            losses = []
    with torch.no_grad():
        val_losses = []
        prediction_votes = {}
        for i, (path, attention_masks, targets, ids) in enumerate(val_loader):
            path = [utterance.to(device, non_blocking=non_blocking) for utterance in path]
            attention_masks = [attention_mask.to(device, non_blocking=non_blocking) for attention_mask in attention_masks]
            targets = targets.to(device, non_blocking=non_blocking)

            with autocast():
                loss = -model(path, attention_masks, targets)
                predictions = model.decode(path, attention_masks)

            val_losses.append(loss.item())

            for id_batch, prediction_batch in zip(ids, predictions):
                for utterance_id, prediction in zip(id_batch, prediction_batch):
                    if utterance_id in prediction_votes:
                        prediction_votes[utterance_id].append(prediction)
                    else:
                        prediction_votes[utterance_id] = [prediction]
        utterance_ids = prediction_votes.keys()
        predictions = np.array([mode(prediction_votes[utterance_id]).mode for utterance_id in utterance_ids])
        labels = [corpus.get_utterance(utterance_id).retrieve_meta('majority_type') for utterance_id in utterance_ids]
        targets = train_dataset.label_encoder.transform(labels)
        accuracy = np.mean(targets == predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(targets, predictions, average='macro')
        print('Validation')
        print(np.mean(val_losses))
        print(accuracy)
        print(precision)
        print(recall)
        print(fscore)
