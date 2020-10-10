from copy import copy
from typing import Dict, List, Tuple

from convokit import Corpus, Utterance, UtteranceNode
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerBase


DELETED_KEYWORD = '[deleted]'
CLASSIFICATION_TOKEN = '[CLS]'


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
    def __init__(self, corpus: Corpus, tokenizer: PreTrainedTokenizerBase, min_len: int=3, max_len: int=12, n_neighbors: int=3, min_to_common_ancestor: int=2) -> None:
        super(ConversationPathDataset, self).__init__()
        self.corpus = corpus
        self.min_len = min_len
        self.max_len = max_len
        self.n_neighbors = n_neighbors
        self.min_to_common_ancestor = min_to_common_ancestor
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
            conversation.initialize_tree_structure()
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
        if len(path) > self.max_len:
            return
        if len(path) >= self.min_len and self.corpus.get_utterance(path[-1]).text != DELETED_KEYWORD and self.corpus.get_utterance(path[-2]).text != DELETED_KEYWORD:
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
        return len(self.idx_to_len)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        conversation_path = self.conversation_paths[index]
        target_utterance_id = conversation_path.utterance_ids[-1]
        conversation_path_neighborhood = [conversation_path] + [self.conversation_paths[self.id_to_idx[neighbor_id]] if neighbor_id in self.id_to_idx else self.id_to_rejects[neighbor_id] for neighbor_id in conversation_path.sample_neighbors(self.n_neighbors)]
        indexed_tokenized_paths = []
        sequence_lengths = []
        for path in conversation_path_neighborhood:
            utterance_ids = path.utterance_ids[:-1] + [target_utterance_id]
            texts = [CLASSIFICATION_TOKEN + self.corpus.get_utterance(utterance_id).text for utterance_id in utterance_ids]
            sequence_lengths.append([len(self.tokenizer.tokenize(text)) for text in texts])
            indexed_tokenized_paths.append([self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding='max_length', truncation=True, return_attention_mask=False)['input_ids'] for text in texts])
        path_tensor = torch.LongTensor(indexed_tokenized_paths)
        sequence_length_tensor = torch.LongTensor(sequence_lengths)
        target_tensor = torch.LongTensor([1] + [0 for i in range(self.n_neighbors)])
        return path_tensor, sequence_length_tensor, target_tensor


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


def conversation_path_collate_fn(batch: Tuple[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor], ...]) -> Tuple[List[torch.LongTensor], torch.LongTensor]:
    path_tensors, sequence_length_tensors, target_tensors = [sample for sample in zip(*batch)]
    batched_path_tensor = torch.stack(path_tensors, 0)
    batched_path_tensor = batched_path_tensor.reshape(-1, batched_path_tensor.shape[2], batched_path_tensor.shape[3])
    batched_sequence_length_tensor = torch.stack(sequence_length_tensors, 0)
    batched_sequence_length_tensor = batched_sequence_length_tensor.reshape(-1, batched_sequence_length_tensor.shape[2])
    batched_target_tensor = torch.stack(target_tensors, 0)
    # dim = 0 and select [0] element of output to get values, not indices
    max_sequence_length_tensor = batched_sequence_length_tensor.max(0)[0]
    batched_utterance_tensors = []
    for utterance_idx, max_sequence_length in enumerate(max_sequence_length_tensor):
        batched_utterance_tensors.append(batched_path_tensor[:, utterance_idx, :max_sequence_length])
    return batched_utterance_tensors, batched_target_tensor
