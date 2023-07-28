# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Callable

import numpy as np
from scipy import special
from scipy.optimize import fminbound

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmDetector():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed

    def aggregate_scores(self, scores: List[List[np.array]], aggregation: str = 'mean') -> List[float]:
        """Aggregate scores along a text."""
        scores = np.asarray(scores)
        if aggregation == 'sum':
           return [ss.sum(axis=0) for ss in scores]
        elif aggregation == 'mean':
            return [ss.mean(axis=0) if ss.shape[0]!=0 else np.ones(shape=(self.vocab_size)) for ss in scores]
        elif aggregation == 'max':
            return [ss.max(axis=0) for ss in scores]
        else:
             raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
        payload_max: int = 0
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt.numpy()[:payload_max+1]
                rts.append(rt)
            score_lists.append(rts)
        return score_lists

    def get_pvalues(
            self, 
            scores: List[np.array], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x ntoks x payload_max
        for ss in scores:
            ntoks = ss.shape[0]
            scores_by_payload = ss.sum(axis=0) if ntoks!=0 else np.zeros(shape=ss.shape[-1]) # payload_max
            pvalues_by_payload = [self.get_pvalue(score, ntoks, eps=eps) for score in scores_by_payload]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues) # bsz x payload_max

    def get_pvalues_by_t(self, scores: List[float]) -> List[float]:
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks)
            pvalues.append(pvalue)
        return pvalues
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError


class MarylandDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The last line shifts the scores by token_id. 
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)

class MarylandDetectorZ(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as MarylandDetector but using zscore """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
        scores[greenlist] = 1
        return scores.roll(-token_id)
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
class OpenaiDetector(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id. 
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)

class OpenaiDetectorZ(WmDetector):

    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as OpenaiDetector but using zscore """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        mu0 = 1
        sigma0 = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
class OpenaiNeymanPearsonDetector(WmDetector):

    def __init__(self, 
            model: LlamaForCausalLM,
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            **kwargs
        ):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.model = model

    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
        payload_max: int = 0
    ) -> np.array: 
        """ 
        Get score increment for each token in list of texts. 
        This version uses the model to get the probabilities, and use them to compute the scores.
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        probs_lists = []
        probs = self.get_probs(torch.tensor(tokens_id).to(self.model.device).long())
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rts = []
            pbs = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                prob = probs[ii][cur_pos-1].cpu().numpy()
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos], prob)
                rts.append(rt.numpy()[:payload_max+1])
                pbs.append(prob)
            score_lists.append(rts)
            probs_lists.append(pbs)
        score_lists = np.asarray(score_lists)
        probs_lists = np.asarray(probs_lists)
        return score_lists, probs_lists

    def get_pvalues(self, scores: np.array, probs: np.array, eps: float=1e-200) -> np.array:
        """Get p-value for each text."""
        pvalues = []
        for ss, pp in zip(scores, probs):
            print(ss, pp)
            scores_by_payload = ss.sum(axis=0) if ss.shape[0]!=0 else np.zeros(shape=1)
            pvalues_by_payload = [self.get_pvalue(score, pp, eps=eps) for score in scores_by_payload]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues)

    def get_pvalues_by_t(self, scores: List[float], probs: List[float], eps: float=1e-200) -> List[float]:
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss, pp in zip(scores, probs):
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, pp[:cum_toks], eps=eps)
            pvalues.append(pvalue)
        return pvalues

    def get_probs(self, input_ids,
                  temperature: float = 0.8, # temperature for sampling
                  top_p: float = 0.95, # top p for sampling
                  ):
        # compute logits
        outputs = self.model.forward(input_ids, return_dict=True)
        logits = outputs['logits']

        # reapply logits processors (temperature & top-p) to get same distribution
        probs = torch.softmax(logits / temperature, dim=-1)

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        mask = mask.scatter(-1, probs_idx, mask)
        probs[mask] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True))

        return torch.gather(probs, 2, input_ids[:,1:,None]).squeeze_(-1)

    def score_tok(self, ngram_tokens, token_id, prob):
        """ rt = -log(1 - rt[xt]]) """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -rs.log().roll(-token_id) * (1/prob - 1)
        return scores

    def get_pvalue(self, score: float, probs: List[float], eps: float):
        # Chernoff bound
        ps = np.asarray(probs)
        k = len(probs)
        ratio = ps / (1 - ps + eps)
        E = (1/(ratio+eps)).sum()
        S = score

        if S > E:
            pvalue = 1.0
        else:
            # to compute p-value we must solve for c*:
            # (1/(c* + ps/(1-ps))).sum() = S
            func = lambda c : (((1 / (c + ratio)).sum() - S)**2).item()
            c1 = (k / S - np.min(ratio)).item()
            c = fminbound(func, 0, c1)

            # upper bound
            pvalue = np.exp(np.sum(-np.log(1 + c / ratio)) + c * S)
        return max(pvalue, eps)
