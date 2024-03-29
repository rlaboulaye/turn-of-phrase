from copy import copy
import math
from typing import Dict, List, Tuple

from convokit import Conversation, Corpus, Utterance, UtteranceNode
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerBase


DELETED_KEYWORD = '[deleted]'
DELTA_STRING = '∆'
ENCODED_DELTA_STRING = '&amp;#8710;'


def text_valid(text: str) -> bool:
    return text is not None and len(text) > 0 and text != DELETED_KEYWORD and not text.isspace()

def select_mask_indices(sequence_length: int, masking_probability: float, max_mask_len: int=3):
    mask_len_probs = np.array([1. / (i + 1) for i in range(max_mask_len)])
    mask_len_probs = mask_len_probs / np.sum(mask_len_probs)
    masked_pos = set()
    while True:
        mask_len = np.random.multinomial(1, mask_len_probs).argmax() + 1
        if len(masked_pos) + mask_len > math.floor(sequence_length * masking_probability):
            break
        # start at 1 to avoid masking CLS and end one early to avoid SEP
        mask_pos = np.random.randint(1, sequence_length - mask_len, 1)
        masked_pos.update(np.arange(mask_pos, mask_pos + mask_len))
    return sorted(list(masked_pos))

def add_title_to_root(corpus: Corpus):
    for conversation in corpus.iter_conversations():
        utterance = corpus.get_utterance(conversation.id)
        title = conversation.retrieve_meta('title')
        if title is None:
            title = ''
        if utterance.text is None:
            utterance.text = title
        else:
            utterance.text = title + ' ' + utterance.text

# filtering code taken directly from https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/datasets/winning-args-corpus/stats.ipynb
def filter_winning_arguments_corpus(corpus: Corpus):
    utterance_ids = corpus.get_utterance_ids()

    #we want the original post made by op, the challenger's comments and all of OP's responses to the challengers
    #these three lists are utterance ids for the original post, challenger comments and op replies respectively

    opPost=[]
    challengerComments=[]
    opReplies=[]
    for iD in utterance_ids:
        
        if corpus.get_utterance(iD).id==corpus.get_utterance(iD).conversation_id:
            opPost.append(iD)
        if corpus.get_utterance(iD).speaker.id != corpus.get_utterance(corpus.get_utterance(iD).conversation_id).speaker.id and corpus.get_utterance(iD).meta['success']==0:
            challengerComments.append(iD)

        if corpus.get_utterance(iD).speaker.id != corpus.get_utterance(corpus.get_utterance(iD).conversation_id).speaker.id and corpus.get_utterance(iD).meta['success']==1:
            challengerComments.append(iD)


        if corpus.get_utterance(iD).id!=corpus.get_utterance(iD).conversation_id and corpus.get_utterance(iD).speaker.id == corpus.get_utterance(corpus.get_utterance(iD).conversation_id).speaker.id and corpus.get_utterance(iD).meta['success']==0:
            opReplies.append(iD)
        if corpus.get_utterance(iD).id!=corpus.get_utterance(iD).conversation_id and corpus.get_utterance(iD).speaker.id == corpus.get_utterance(corpus.get_utterance(iD).conversation_id).speaker.id and corpus.get_utterance(iD).meta['success']==1:
            opReplies.append(iD)
            
    #subset challenger and op replies for later use (into successful and unsuccessful arguments)
    challengerPos=[]
    challengerNeg=[]
    for iD in challengerComments:
        if corpus.get_utterance(iD).meta['success']==1:
            challengerPos.append(iD)
        if corpus.get_utterance(iD).meta['success']==0:
            challengerNeg.append(iD)

    #these are OP's replies to successful and unsuccessful challengers        
    opReplyPos=[]
    opReplyNeg=[]
    for iD in opReplies:
        if corpus.get_utterance(iD).meta['success']==1:
            opReplyPos.append(iD)
        if corpus.get_utterance(iD).meta['success']==0:
            opReplyNeg.append(iD)

    subset=opPost+challengerComments+opReplies

    #collect utterance dict given the subset of ids
    utterance_list=[]
    for iD in subset:
        utterance_list.append(corpus.get_utterance(iD))

    #this subset separates OP comments and challenger utterances from all other comments in every conversation (thread)
    corpus = Corpus(utterances=utterance_list)

    return corpus


class ConversationPath():
    """docstring for ConversationPath"""
    def __init__(self, utterance_ids: List[int]) -> None:
        super(ConversationPath, self).__init__()
        self.id = utterance_ids[-1]
        self.utterance_ids = utterance_ids
        self.neighbor_ids = []

    def sample_neighbors(self, n: int) -> List[int]:
        permuted_indices = np.random.permutation(len(self.neighbor_ids))
        return self.neighbor_ids[:n]


class ConversationPathDataset(Dataset):
    """docstring for ConversationPathDataset"""
    def __init__(self, corpus: Corpus, tokenizer: PreTrainedTokenizerBase, min_len: int=3, max_len: int=6, n_neighbors: int=1, min_to_common_ancestor: int=2, max_tokenization_len: int=4096, masking_probability: float=.15) -> None:
        super(ConversationPathDataset, self).__init__()
        self.corpus = corpus
        self.min_len = min_len
        self.max_len = max_len
        self.n_neighbors = n_neighbors
        self.min_to_common_ancestor = min_to_common_ancestor
        self.max_tokenization_len = max_tokenization_len
        self.masking_probability = masking_probability
        self.tokenizer = tokenizer
        self._initialize_conversation_paths()

    def _initialize_conversation_paths(self) -> None:
        self.conversation_paths = []
        self.id_to_idx = {}
        # need a way to access paths that had insufficient neighbors, but are a different path's neighbor
        self.id_to_rejects = {}
        self.conversation_indices_by_len = [[] for i in range(self.max_len - self.min_len + 1)]
        for conversation in self.corpus.iter_conversations():
            candidates_by_len = [{} for i in range(self.max_len - self.min_len + 1)]
            # skip conversations with reply-to chains that do not form a valid tree
            try:
                conversation.initialize_tree_structure()
            except ValueError:
                continue
            root = conversation.tree
            self._dfs_conversation_path_traversal(root, [root.utt.id], candidates_by_len)
            for candidates, conversation_indices in zip(candidates_by_len, self.conversation_indices_by_len):
                for candidate in candidates.values():
                    if len(candidate.neighbor_ids) > self.n_neighbors:
                        conversation_indices.append(len(self.conversation_paths))
                        self.id_to_idx[candidate.id] = len(self.conversation_paths)
                        self.conversation_paths.append(candidate)
                    else:
                        self.id_to_rejects[candidate.id] = candidate

    def _dfs_conversation_path_traversal(self, utterance_node: UtteranceNode, path: List[int], candidates_by_len: List[Dict[str, ConversationPath]]) -> None:
        path.append(utterance_node.utt.id)
        if len(path) > self.max_len or not text_valid(utterance_node.utt.text):
            return
        # if len(path) >= self.min_len and text_valid(self.corpus.get_utterance(path[-1]).text) and text_valid(self.corpus.get_utterance(path[-2]).text):
        if len(path) >= self.min_len:
            conversation_path = ConversationPath(path)
            candidates = candidates_by_len[len(path) - self.min_len]
            for candidate_path in candidates.values():
                if self._is_unique_neighbor(conversation_path, candidate_path, candidates):
                    conversation_path.neighbor_ids.append(candidate_path.id)
                    candidate_path.neighbor_ids.append(conversation_path.id)
            candidates_by_len[len(path) - self.min_len][conversation_path.id] = conversation_path
        for child in utterance_node.children:
            self._dfs_conversation_path_traversal(child, copy(path), candidates_by_len)

    def _is_unique_neighbor(self, conversation_path: ConversationPath, candidate_path: ConversationPath, candidates: Dict[str, ConversationPath]) -> bool:
        if self._distance_to_common_ancestor(conversation_path, candidate_path) < self.min_to_common_ancestor:
            return False
        for neighbor_id in conversation_path.neighbor_ids:
            neighbor_path = candidates[neighbor_id]
            if self._distance_to_common_ancestor(neighbor_path, candidate_path) < self.min_to_common_ancestor:
                return False
        return True

    def _distance_to_common_ancestor(self, conversation_path_1: ConversationPath, conversation_path_2: ConversationPath):
        for i, (id_1, id_2) in enumerate(zip(reversed(conversation_path_1.utterance_ids), reversed(conversation_path_2.utterance_ids))):
            if id_1 == id_2:
                return i

    def get_indices_by_len(self) -> Dict[int, List[int]]:
        return {i + self.min_len: self.conversation_indices_by_len[i] for i in range(len(self.conversation_indices_by_len))}

    def __len__(self) -> int:
        return len(self.conversation_paths)

    # def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, List[List[int]]]:
    #     conversation_path = self.conversation_paths[index]
    #     target_utterance_id = conversation_path.utterance_ids[-1]
    #     conversation_path_neighborhood = [conversation_path] + [self.conversation_paths[self.id_to_idx[neighbor_id]] if neighbor_id in self.id_to_idx else self.id_to_rejects[neighbor_id] for neighbor_id in conversation_path.sample_neighbors(self.n_neighbors)]
    #     tokenized_paths = []
    #     attention_masks = []
    #     sequence_lengths = []
    #     for path in conversation_path_neighborhood:
    #         utterance_ids = path.utterance_ids[:-1] + [target_utterance_id]
    #         texts = [self.corpus.get_utterance(utterance_id).text for utterance_id in utterance_ids]
    #         sequence_lengths.append([len(self.tokenizer.tokenize(text)) for text in texts])
    #         encodings = [self.tokenizer(text, max_length=self.max_tokenization_len, padding='max_length', truncation=True) for text in texts]
    #         tokenized_paths.append([encoding['input_ids'] for encoding in encodings])
    #         attention_masks.append([np.array(encoding['attention_mask']) for encoding in encodings])
    #         for sequence_length, attention_mask in zip(sequence_lengths[-1], attention_masks[-1]):
    #             mask_indices = select_mask_indices(min(sequence_length, self.max_tokenization_len), self.masking_probability)
    #             attention_mask[mask_indices] = 0
    #     path_tensor = torch.LongTensor(tokenized_paths)
    #     attention_mask_tensor = torch.LongTensor(attention_masks)
    #     sequence_length_tensor = torch.LongTensor(sequence_lengths)
    #     target_tensor = torch.LongTensor([0])
    #     return path_tensor, attention_mask_tensor, sequence_length_tensor, target_tensor, path

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor, List[List[int]]]:
        conversation_path = self.conversation_paths[index]
        target_utterance_id = conversation_path.utterance_ids[-1]
        conversation_path_neighborhood = [conversation_path] + [self.conversation_paths[self.id_to_idx[neighbor_id]] if neighbor_id in self.id_to_idx else self.id_to_rejects[neighbor_id] for neighbor_id in conversation_path.sample_neighbors(self.n_neighbors)]

        conversation_path_neighborhood_texts = []
        for path in conversation_path_neighborhood:
            utterance_ids = path.utterance_ids[:-1] + [target_utterance_id]
            texts = [self.corpus.get_utterance(utterance_id).text for utterance_id in utterance_ids]
            texts = [' '.join([token[1:] for token in self.tokenizer.tokenize(text)][:self.max_tokenization_len // self.max_len]) for text in texts]
            path_text = self.tokenizer.sep_token.join(texts)
            conversation_path_neighborhood_texts.append(path_text)
        tokenized_path = self.tokenizer(conversation_path_neighborhood_texts, max_length=self.max_tokenization_len, padding=True, truncation=True, return_tensors='pt')
        return tokenized_path.input_ids, tokenized_path.attention_mask, torch.LongTensor([0]), path


def conversation_path_collate_fn(batch: Tuple[Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor], ...]) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor], torch.LongTensor]:
    path_tensors, attention_masks, target_tensors, ids = [sample for sample in zip(*batch)]
    max_path_tensor_len = max([path_tensor.shape[-1] for path_tensor in path_tensors])
    padded_path_tensors = [torch.nn.functional.pad(path_tensor, (0, max_path_tensor_len - path_tensor.shape[-1])) for path_tensor in path_tensors]
    batched_path_tensor = torch.stack(padded_path_tensors, 0)
    padded_attention_masks = [torch.nn.functional.pad(attention_mask, (0, max_path_tensor_len - attention_mask.shape[-1])) for attention_mask in attention_masks]
    batched_attention_mask = torch.stack(padded_attention_masks, 0)

    # batched_attention_mask_tensor = torch.stack(attention_mask_tensor, 0)
    # batched_attention_mask_tensor = batched_attention_mask_tensor.reshape(-1, batched_attention_mask_tensor.shape[2], batched_attention_mask_tensor.shape[3])

    batched_target_tensor = torch.cat(target_tensors, 0)

    return batched_path_tensor, batched_attention_mask, batched_target_tensor, ids


class CoarseDiscourseDataset(Dataset):
    """docstring for CoarseDiscourseDataset"""
    def __init__(self, corpus: Corpus, conversations: List[Conversation], tokenizer: PreTrainedTokenizerBase, max_len: int=8, max_tokenization_len: int=256) -> None:
        super(CoarseDiscourseDataset, self).__init__()
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_tokenization_len = max_tokenization_len
        labels = set()
        # make sure that no utterance has None timestamp to prepare for initializing of tree structure
        for utterance in self.corpus.iter_utterances():
            if utterance.timestamp is None:
                utterance.timestamp = 0
            label = utterance.retrieve_meta('majority_type')
            if label is not None:
                labels.add(label)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(labels))
        self.paths = []
        self.indices_by_len = [[] for i in range(self.max_len)]
        for conversation in conversations:
            try:
                paths = []
                for path in conversation.get_root_to_leaf_paths():
                    id_path = [utterance.id for utterance in path]
                    cutoff = len(id_path)
                    for i, utterance_id in enumerate(id_path):
                        utterance = self.corpus.get_utterance(utterance_id)
                        if utterance.retrieve_meta('majority_type') is None or utterance.retrieve_meta('majority_type') == 'other' or not text_valid(utterance.text):
                            cutoff = i
                            break
                    if cutoff == 0:
                        continue
                    id_path = id_path[:cutoff]
                    if np.any([path == id_path for path in paths]):
                        continue
                    paths.append(id_path)
                    if len(id_path) <= self.max_len:
                        self.indices_by_len[len(id_path) - 1].append(len(self.paths))
                        self.paths.append(id_path)
                    else:
                        for i in range(len(id_path) + 1 - self.max_len):
                            truncated_id_path = id_path[i: self.max_len + i]
                            self.indices_by_len[self.max_len - 1].append(len(self.paths))
                            self.paths.append(truncated_id_path)
            except:
                continue

    def get_indices_by_len(self) -> Dict[int, List[int]]:
        return {i + 1: self.indices_by_len[i] for i in range(len(self.indices_by_len))}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        path = self.paths[index]
        texts = [self.corpus.get_utterance(utterance_id).text for utterance_id in path]
        texts = [' '.join([token[1:] for token in self.tokenizer.tokenize(text)][:self.max_tokenization_len // self.max_len]) for text in texts]
        path_text = self.tokenizer.sep_token.join(texts)
        tokenized_path = self.tokenizer(path_text, max_length=self.max_tokenization_len, truncation=True, return_tensors='pt')
        labels = [self.corpus.get_utterance(utterance_id).retrieve_meta('majority_type') for utterance_id in path]
        targets = torch.LongTensor(self.label_encoder.transform(labels))
        return tokenized_path.input_ids, tokenized_path.attention_mask, targets, path


class WinningArgumentsDataset(Dataset):
    """docstring for WinningArgumentsDataset"""
    def __init__(self, corpus: Corpus, conversations: List[Conversation], tokenizer: PreTrainedTokenizerBase, max_len: int=8, max_tokenization_len: int=256) -> None:
        super(WinningArgumentsDataset, self).__init__()
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_tokenization_len = max_tokenization_len
        self.paths = []
        self.indices_by_len = [[] for i in range(self.max_len)]
        for conversation in conversations:
            try:
                for path in conversation.get_root_to_leaf_paths():
                    id_path = [utterance.id for utterance in path]
                    # remove all utterances including and past the awarded delta
                    for i, utterance_id in enumerate(id_path):
                        utterance = self.corpus.get_utterance(utterance_id)
                        if (DELTA_STRING in utterance.text or ENCODED_DELTA_STRING in utterance.text) and (i != 0 and i != 1):
                            id_path = id_path[:i]
                            break
                    # truncate path to max
                    id_path = id_path[:self.max_len]
                    self.indices_by_len[len(id_path) - 1].append(len(self.paths))
                    self.paths.append(id_path)
            except:
                continue

    def get_indices_by_len(self) -> Dict[int, List[int]]:
        return {i + 1: self.indices_by_len[i] for i in range(len(self.indices_by_len))}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        path = self.paths[index]
        texts = [self.corpus.get_utterance(utterance_id).text for utterance_id in path]
        texts = [' '.join([token[1:] for token in self.tokenizer.tokenize(text)][:self.max_tokenization_len // self.max_len]) for text in texts]
        path_text = self.tokenizer.sep_token.join(texts)
        tokenized_path = self.tokenizer(path_text, max_length=self.max_tokenization_len, truncation=True, return_tensors='pt')
        targets = torch.FloatTensor([self.corpus.get_utterance(path[-1]).retrieve_meta('success')])
        return tokenized_path.input_ids, tokenized_path.attention_mask, targets, path


class ConversationsGoneAwryDataset(Dataset):
    """docstring for ConversationsGoneAwryDataset"""
    def __init__(self, corpus: Corpus, conversations: List[Conversation], tokenizer: PreTrainedTokenizerBase, max_len: int=8, max_tokenization_len: int=256) -> None:
        super(ConversationsGoneAwryDataset, self).__init__()
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_tokenization_len = max_tokenization_len
        self.paths = []
        self.indices_by_len = [[] for i in range(self.max_len)]
        for conversation in conversations:
            path = conversation.get_root_to_leaf_paths()[0]
            id_path = [utterance.id for utterance in path]
            # truncate path to max
            id_path = id_path[:self.max_len]
            self.indices_by_len[len(id_path) - 1].append(len(self.paths))
            self.paths.append(id_path)

    def get_indices_by_len(self) -> Dict[int, List[int]]:
        return {i + 1: self.indices_by_len[i] for i in range(len(self.indices_by_len))}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        path = self.paths[index]
        texts = [self.corpus.get_utterance(utterance_id).text for utterance_id in path]
        texts = [' '.join([token[1:] for token in self.tokenizer.tokenize(text)][:self.max_tokenization_len // self.max_len]) for text in texts]
        path_text = self.tokenizer.sep_token.join(texts)
        tokenized_path = self.tokenizer(path_text, max_length=self.max_tokenization_len, truncation=True, return_tensors='pt')
        targets = torch.FloatTensor([self.corpus.get_conversation(path[0]).retrieve_meta('has_removed_comment')])
        return tokenized_path.input_ids, tokenized_path.attention_mask, targets, path


class PolitenessDataset(Dataset):
    """docstring for PolitenessDataset"""
    def __init__(self, utterances: List[Utterance], tokenizer: PreTrainedTokenizerBase, max_tokenization_len=256):
        super(PolitenessDataset, self).__init__()
        self.utterances = utterances
        self.tokenizer = tokenizer
        self.max_tokenization_len = max_tokenization_len

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        utterance = self.utterances[index]
        encoding = self.tokenizer(utterance.text, max_length=self.max_tokenization_len, padding='max_length', truncation=True)
        path_tensor = torch.LongTensor(encoding['input_ids'])
        attention_mask_tensor = torch.LongTensor(encoding['attention_mask'])
        if utterance.retrieve_meta('Binary') == -1:
            target_tensor = torch.LongTensor([0])
        elif utterance.retrieve_meta('Binary') == 1:
            target_tensor = torch.LongTensor([1])
        return path_tensor, attention_mask_tensor, target_tensor


class ConversationPathBatchSampler(Sampler):
    """docstring for ConversationPathBatchSampler"""
    def __init__(self, batch_size: int, min_len: int, indices_by_len: Dict[int, List[int]]) -> None:
        self.batch_size = batch_size
        self.min_len = min_len
        self.indices_by_len = {len_key: np.array(indices_by_len[len_key]) for len_key in indices_by_len}

    def __iter__(self) -> List[int]:
        indices_by_len = copy(self.indices_by_len)
        num_remaining_indices = sum([len(indices_by_len[len_key]) for len_key in indices_by_len])
        while num_remaining_indices > 0:
            sampling_weights = [len(indices_by_len[len_key]) / num_remaining_indices for len_key in indices_by_len]
            sampled_len = np.random.multinomial(1, sampling_weights).argmax() + self.min_len
            indices = indices_by_len[sampled_len]
            permuted_indices = indices[np.random.permutation(len(indices))]
            yield permuted_indices[:self.batch_size]
            if len(permuted_indices) <= self.batch_size:
                indices_by_len[sampled_len] = np.array([], dtype=np.int)
            else:
                indices_by_len[sampled_len] = permuted_indices[self.batch_size:]
            num_remaining_indices = sum([len(indices_by_len[len_key]) for len_key in indices_by_len])

    def __len__(self) -> int:
        return sum([len(self.indices_by_len[len_key]) for len_key in self.indices_by_len])


# def conversation_path_collate_fn(batch: Tuple[Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor], ...]) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor], torch.LongTensor]:
#     path_tensors, attention_mask_tensor, sequence_length_tensors, target_tensors, ids = [sample for sample in zip(*batch)]
#     batched_path_tensor = torch.stack(path_tensors, 0)
#     batched_path_tensor = batched_path_tensor.reshape(-1, batched_path_tensor.shape[2], batched_path_tensor.shape[3])
#     batched_attention_mask_tensor = torch.stack(attention_mask_tensor, 0)
#     batched_attention_mask_tensor = batched_attention_mask_tensor.reshape(-1, batched_attention_mask_tensor.shape[2], batched_attention_mask_tensor.shape[3])
#     batched_sequence_length_tensor = torch.stack(sequence_length_tensors, 0)
#     batched_sequence_length_tensor = batched_sequence_length_tensor.reshape(-1, batched_sequence_length_tensor.shape[2])
#     batched_target_tensor = torch.cat(target_tensors, 0)
#     # dim = 0 and select [0] element of output to get values, not indices
#     max_sequence_length_tensor = batched_sequence_length_tensor.max(0)[0]
#     batched_utterance_tensors = []
#     batched_attention_mask_tensors = []
#     for utterance_idx, max_sequence_length in enumerate(max_sequence_length_tensor):
#         batched_utterance_tensors.append(batched_path_tensor[:, utterance_idx, :max_sequence_length])
#         batched_attention_mask_tensors.append(batched_attention_mask_tensor[:, utterance_idx, :max_sequence_length])
#     return batched_utterance_tensors, batched_attention_mask_tensors, batched_target_tensor, ids
