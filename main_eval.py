# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
python main_eval.py \
    --json_path examples/results.jsonl --text_key result \
    --json_path_ori data/alpaca_data.json --text_key_ori output \
    --do_wmeval True --method openai --seeding hash --ngram 2 --scoring_method v2 \
    --payload 0 --payload_max 4 \
    --output_dir output/ 
"""

import argparse
from typing import List
import os
import json

import tqdm
import pandas as pd
import numpy as np

import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from wm import (OpenaiDetector, OpenaiDetectorZ, OpenaiNeymanPearsonDetector, 
                MarylandDetector, MarylandDetectorZ)
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--text_key', type=str, default='text',
                        help='key to access text in json dict')
    parser.add_argument('--tokenizer_dir', type=str, default='llama-7b')
    parser.add_argument('--json_path_ori', type=str, default=None,
                        help='path to json file containing original text to compute sbert score with (optional). \
                              If not provided, sbert score will not be computed')
    parser.add_argument('--text_key_ori', type=str, default='result',
                        help='key to access text in json dict')

    # watermark parameters
    parser.add_argument('--do_wmeval', type=utils.bool_inst, default=True,
                        help='whether to do watermark evaluation')
    parser.add_argument('--method', type=str, default='none',
                        help='watermark detection method')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')


    # multibit
    parser.add_argument('--payload', type=int, default=0, 
                        help='message')
    parser.add_argument('--payload_max', type=int, default=0, 
                        help='maximal message')
    
    # attack
    parser.add_argument('--attack_name', type=str, default='none',
                        help='attack name to be applied to text before evaluation. Choose between: \
                        none (no attack), tok_substitution (randomly substitute tokens)')
    parser.add_argument('--attack_param', type=float, default=0,
                        help='attack parameter. For tok_substitution, it is the probability of substitution')

    # useless
    parser.add_argument('--delta', type=float, default=2.0, 
                        help='delta for maryland (useless for detection)')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='temperature for generation (useless for detection)')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to evaluate, if None, take all texts')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--do_eval', type=utils.bool_inst, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None,
                        help='split the texts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat texts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat texts as a whole')


    return parser

def load_results(json_path: str, nsamples: int=None, text_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    new_prompts = [o[text_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    return new_prompts

def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # build watermark detector
    if args.method == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method == "openaiz":
        detector = OpenaiDetectorZ(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
    elif args.method == "marylandz":
        detector = MarylandDetectorZ(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
    elif args.method == "openainp":
        # build model
        if args.model_name == "llama-7b":
            model_name = "llama-7b"
            adapters_name = None
        if args.model_name == "guanaco-7b":
            model_name = "llama-7b"
            adapters_name = 'timdettmers/guanaco-7b'
        elif args.model_name == "guanaco-13b":
            model_name = "llama-13b"
            adapters_name = 'timdettmers/guanaco-13b'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        args.ngpus = torch.cuda.device_count() if args.ngpus is None else args.ngpus
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={i: '32000MB' for i in range(args.ngpus)},
            offload_folder="offload",
        )
        if adapters_name is not None:
            model = PeftModel.from_pretrained(model, adapters_name)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"Using {args.ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")
        detector = OpenaiNeymanPearsonDetector(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)

    # load results and (optional) do splits
    results = load_results(json_path=args.json_path, text_key=args.text_key, nsamples=args.nsamples)
    print(f"Loaded {len(results)} results.")
    if args.split is not None:
        nresults = len(results)
        left = nresults * args.split // args.nsplits 
        right = nresults * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nresults
        results = results[left:right]
        print(f"Creating results from {left} to {right}")

    # attack results
    attack_name = args.attack_name
    attack_param = args.attack_param
    if attack_name == 'tok_substitution':
        def attack(text):
            tokens_id = tokenizer.encode(text, add_special_tokens=False)
            for token_id in tokens_id:
                if np.random.rand() < attack_param:
                    tokens_id[tokens_id.index(token_id)] = np.random.randint(tokenizer.vocab_size)
            new_text = tokenizer.decode(tokens_id)
            return new_text
        if args.attack_param > 0:
            print(results[:2])
            results = [attack(text) for text in results]
            print(f"Attacked results with {attack_name}({attack_param})")
            print(results[:2])

    # evaluate
    log_stats = []
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'scores.jsonl'), 'w') as f:

        # build sbert model
        if args.json_path_ori is not None:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            results_orig = load_results(json_path=args.json_path_ori, nsamples=args.nsamples, text_key=args.text_key_ori)
            if args.split is not None:
                results_orig = results_orig[left:right]

        for ii, text in tqdm.tqdm(enumerate(results), total=len(results)):
            log_stat = {
                'text_index': ii,
            }
            if args.do_wmeval:
                # compute watermark score
                if args.method != "openainp":
                    scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method, payload_max=args.payload_max)
                    scores = detector.aggregate_scores(scores_no_aggreg) # p 1
                    pvalues = detector.get_pvalues(scores_no_aggreg) 
                else:
                    scores_no_aggreg, probs = detector.get_scores_by_t([text], scoring_method=args.scoring_method, payload_max=args.payload_max)
                    scores = detector.aggregate_scores(scores_no_aggreg) # p 1
                    pvalues = detector.get_pvalues(scores_no_aggreg, probs)
                if args.payload_max:
                    # decode payload and adjust pvalues
                    payloads = np.argmin(pvalues, axis=1).tolist()
                    pvalues = pvalues[:,payloads][0].tolist() # in fact pvalue is of size 1, but the format could be adapted to take multiple text at the same time
                    scores = [float(s[payload]) for s,payload in zip(scores,payloads)]
                    # adjust pvalue to take into account the number of tests (2**payload_max)
                    # use exact formula for high values and (more stable) upper bound for lower values
                    M = args.payload_max+1
                    pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
                else:
                    payloads = [ 0 ] * len(pvalues)
                    pvalues = pvalues[:,0].tolist()
                    scores = [float(s[0]) for s in scores]
                num_tokens = [len(score_no_aggreg) for score_no_aggreg in scores_no_aggreg]
                log_stat['num_token'] =  num_tokens[0]
                log_stat['score'] =  scores[0]
                log_stat['pvalue'] =  pvalues[0]
                log_stat['log10_pvalue'] = float(np.log10(log_stat['pvalue']))
            if args.json_path_ori is not None:
                # compute sbert score
                text_orig = results_orig[ii]
                xs = sbert_model.encode([text, text_orig], convert_to_tensor=True)
                score_sbert = cossim(xs[0], xs[1]).item()
                log_stat['score_sbert'] = score_sbert
            log_stats.append(log_stat)
            f.write('\n' + json.dumps(log_stat))
        df = pd.DataFrame(log_stats)
        print(f">>> Scores: \n{df.describe(percentiles=[])}")
        print(f"Saved scores to {os.path.join(args.output_dir, 'scores.csv')}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
