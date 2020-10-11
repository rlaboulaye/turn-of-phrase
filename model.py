from typing import List

import torch
from torch import nn
from transformers import PreTrainedModel


class HierarchicalRNN(nn.Module):
    """docstring for HierarchicalRNN"""
    def __init__(self, utterance_encoder: PreTrainedModel, conversation_encoder: nn.RNNBase, n_classes: int=None) -> None:
        super(HierarchicalRNN, self).__init__()
        self.utterance_encoder = utterance_encoder
        self.conversation_encoder = conversation_encoder
        if n_classes is None:
            self.classifier = None
        else:
            self.classifier = nn.Linear(in_features=conversation_encoder.hidden_size, out_features=n_classes)

    def forward(self, conversation_tokens: List[torch.LongTensor], conversation_attention_mask: List[torch.FloatTensor]):
        encoded_utterances = []
        for utterance_tokens, utterance_mask in zip(conversation_tokens, conversation_attention_mask):
            encoded_utterances.append(self.utterance_encoder(utterance_tokens, utterance_mask)[1])
        encoded_conversation = torch.stack(encoded_utterances, 0)
        output, state_n = self.conversation_encoder(encoded_conversation)
        if self.classifier is None:
            return output, state_n
        else:
            # [0] element of state is the hidden state ([1] for lstm is cell state)
            # [-1] select the top layer
            return self.classifier(state_n[0][-1])
