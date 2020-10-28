from typing import List

import torch
from torch import nn
from crf import CRF
from transformers import PreTrainedModel


class HierarchicalRNN(nn.Module):
    """docstring for HierarchicalRNN"""
    def __init__(self, utterance_encoder: PreTrainedModel, conversation_encoder: nn.RNNBase) -> None:
        super(HierarchicalRNN, self).__init__()
        self.utterance_encoder = utterance_encoder
        self.conversation_encoder = conversation_encoder

    def forward(self, conversation_tokens: List[torch.LongTensor], conversation_attention_mask: List[torch.FloatTensor]):
        encoded_utterances = []
        for utterance_tokens, utterance_mask in zip(conversation_tokens, conversation_attention_mask):
            encoded_utterances.append(self.utterance_encoder(utterance_tokens, utterance_mask)[1])
        encoded_conversation = torch.stack(encoded_utterances, 0)
        output, state_n = self.conversation_encoder(encoded_conversation)
        return output, state_n

class ConversationClassificationHRNN(nn.Module):
    """docstring for ConversationClassificationHRNN"""
    def __init__(self, utterance_encoder: PreTrainedModel, conversation_encoder: nn.RNNBase, n_classes: int) -> None:
        super(ConversationClassificationHRNN, self).__init__()
        self.hrnn = HierarchicalRNN(utterance_encoder, conversation_encoder)
        self.output_layer = nn.Linear(in_features=conversation_encoder.hidden_size, out_features=n_classes)

    def forward(self, conversation_tokens: List[torch.LongTensor], conversation_attention_mask: List[torch.FloatTensor]):
        output, state_n = self.hrnn(conversation_tokens, conversation_attention_mask)
        # [-1] selects the representation for the last element in the sequence
        return self.output_layer(output[-1])

class UtteranceClassificationHRNN(nn.Module):
    """docstring for UtteranceClassificationHRNN"""
    def __init__(self, utterance_encoder: PreTrainedModel, conversation_encoder: nn.RNNBase, n_classes: int) -> None:
        super(UtteranceClassificationHRNN, self).__init__()
        self.hrnn = HierarchicalRNN(utterance_encoder, conversation_encoder)
        self.output_layer = nn.Linear(in_features=conversation_encoder.hidden_size, out_features=n_classes)
        self.crf = CRF(n_classes)

    def forward(self, conversation_tokens: List[torch.LongTensor], conversation_attention_mask: List[torch.FloatTensor], labels: torch.LongTensor):
        output, state_n = self.hrnn(conversation_tokens, conversation_attention_mask)
        emissions = self.output_layer(output.reshape(-1, output.shape[2])).reshape(output.shape[0], output.shape[1], -1)
        return self.crf(emissions, labels.T)

    def decode(self, conversation_tokens: List[torch.LongTensor], conversation_attention_mask: List[torch.FloatTensor]):
        output, state_n = self.hrnn(conversation_tokens, conversation_attention_mask)
        emissions = self.output_layer(output.reshape(-1, output.shape[2])).reshape(output.shape[0], output.shape[1], -1)
        return self.crf.decode(emissions)
