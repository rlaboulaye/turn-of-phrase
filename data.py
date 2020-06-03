from itertools import chain
from typing import Tuple

from convokit import Corpus
import torch
from torch import LongTensor
from torch.nn import Module
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


def conversation_collate_fn(batch: Tuple[Tuple[LongTensor, ...], ...]) -> Tuple[LongTensor, ...]:
    inputs = [data[0] for data in batch]
    flattened_inputs = list(chain.from_iterable([list(input_tensor) for input_tensor in inputs]))
    flattened_word_padded_inputs = pad_sequence(flattened_inputs, batch_first=True)
    word_padded_inputs = torch.split(flattened_word_padded_inputs, [input_tensor.shape[0] for input_tensor in inputs])
    utterance_padded_inputs = pad_sequence(word_padded_inputs, batch_first=False)
    batched_data = [utterance_padded_inputs]
    padding_value = -2
    for i in range(1, len(batch[0])):
        labels = [data[i] for data in batch]
        padded_labels = pad_sequence(labels, padding_value=padding_value, batch_first=False)
        batched_data.append(padded_labels)
    return tuple(batched_data)


class ConversationDataset(Dataset):

    def __init__(self, corpus: Corpus, utterance_label: str = None, conversation_label: str = None, max_sequence_length: int = 512) -> None:
        self.corpus = corpus
        self.conversation_ids = corpus.get_conversation_ids()
        self.utterance_label = utterance_label
        self.conversation_label = conversation_label
        self.max_sequence_length = max_sequence_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self) -> int:
        return len(self.conversation_ids)

    def __getitem__(self, idx: int) -> Tuple[LongTensor, ...]:
        conversation = self.corpus.get_conversation(self.conversation_ids[idx])
        utterances = []
        reply_to_indices = []
        utterance_indices = {}
        for i, utterance in enumerate(conversation.traverse('bfs')):
            text = utterance.text
            # if the utterance is the first in a conversation, add the title text to the utterance text
            if utterance._id == utterance.root:
                title = utterance.get_conversation().get_info('title')
                text = '\n'.join([title, text])
            # transform the text into a list of tokens
            tokenized_text = self.tokenizer.tokenize(text)
            # truncate the text by the max sequence length
            tokenized_text = tokenized_text[:self.max_sequence_length]
            # transform the list of tokens into a list of token ids
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # transform the list of tokens into a tensor and append it
            utterances.append(LongTensor(indexed_tokens))
            # retrieve the list index of the utterance being replied to
            # set to -1 if there is no utterance being replied to
            if utterance.reply_to is None:
                reply_to_indices.append(-1)
            else:
                reply_to_index = utterance_indices[utterance.reply_to]
                reply_to_indices.append(reply_to_index)
            # store the utterance's id and position in utterances as a key value pair
            utterance_indices[utterance._id] = i
        # transform the utterances and reply_to_indices into tensors
        utterances_tensor = pad_sequence(utterances, batch_first=True)
        reply_to_indices_tensor = LongTensor(reply_to_indices)
        return utterances_tensor, reply_to_indices_tensor
