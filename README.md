# 🧱 Three Bricks to Consolidate Watermarks for LLMs

This repository allows to generate and watermark text using various watermarking methods, with LLaMA models.
Detection of the watermarks is possible using various statistical tests, such as the ones introduced in the paper.

For additional technical details, see [**the paper**](https://arxiv.org/abs/2308.00113).  


[[`Webpage`](https://pierrefdz.github.io/publications/threebricks/)]
[[`arXiv`](https://arxiv.org/abs/2308.00113)]
[[`Demo`](https://huggingface.co/spaces/NohTow/LLM_watermarking)]
[[`Demo Llama 2`]](https://huggingface.co/spaces/NohTow/Llama2_watermarking)


## Setup


### Requirements

First, clone the repository locally and move inside the folder:
```cmd
git clone https://github.com/facebookresearch/three_bricks
cd three_bricks
```
To install the main dependencies, we recommand using conda, and install the remaining dependencies with pip:
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
This codebase has been developed with python version 3.11, PyTorch version 2.0.1, CUDA 11.7.

### Data

The paper uses prompts from the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
They can be downloaded with:
```cmd
mkdir data
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -P data/
```

### Models

We use the [HuggingFace](https://huggingface.co/) library to load the models.
To download the models, please see https://huggingface.co/docs/transformers/main/model_doc/llama.


## Usage


### Watermarked LLM generation + evaluation

The repository supports different watermarking methods, including the works of [Kirchenbauer et al.](https://arxiv.org/abs/2301.10226) and of [Aaronson et al](https://www.scottaaronson.com/talks/watermark.ppt).
The detection can be performed using the original statistical tests, or using the ones introduced in the paper.

You can use the script by running the `main_watermark.py` file with the appropriate command-line arguments.
The main ones are:
- `--model_name`: The name of the pre-trained model to use for text generation and analysis. Supported model names include "llama-7b", "guanaco-7b" and "guanaco-13b".
- `--prompt_path`: The path to the JSON file containing prompts. Default value: "data/alpaca_data.json."
- `--method`: Choose a watermarking method for text generation. Options: "none" (no watermarking), "openai" (Aaronson et al.), "maryland" (Kirchenbauer et al.). Default value: "none."
- `--method_detect`: Choose a statistical test to detect watermark. "same" uses the grounded statistical test with the same method as for generation, "openaiz" and "marylandz" use the z-tests, and "openainp" uses the Neyman-Pearson test. Default value: "same."
- `--scoring_method`: Method for scoring tokens. Options: "none" (score every token), "v1" (score token when the watermark context is unique), "v2" (score token when {wm context + token} is unique). Default value: "none."
- `--ngram`: Watermark context width for RNG key generation. Default value: 4.
- `--payload`: Message for multi-bit watermarking. It must be inferior to `--payload_max`. Default value: 0
- `--payload_max`: Maximal message for multi-bit watermarking. It must be inferior to the vocabulary size. Must be >0 to do multi-bit watermarking. Default value: 0.

For example, the following command generates watermarked text using the method of Aaronson et al., and detects the watermark using the statistical test of the paper.
```cmd
python main_watermark.py --model_name guanaco-7b \
    --prompt_type guanaco --prompt_path data/alpaca_data.json --nsamples 10 --batch_size 16 \
    --method openai --temperature 1.0 --seeding hash --ngram 2 --method_detect openai --scoring_method v2 \
    --payload 0 --payload_max 4 \
    --output_dir output/
```

#### Output

The previous script generates watermarked text and saves the results in the specified output directory. The output files include:

**`results.jsonl`.** Contains the generated watermarked text for each prompt in JSONL format. 

**`scores.jsonl`.** Contains the analysis results for each watermarked text in JSONL format.
In particular, the `scores.jsonl` file contains the following fields:

| Field | Description |
| --- | --- |
| `text_index` | Index of the prompt in the JSON file |
| `num_token` | Number of analyzed tokens in the text |
| `score` | Watermark score of the text |
| `pvalue` | p-value of the detection test |
| `score_sbert` | Cosine similarity score between watermarked completion and ground truth answer |
| `all_pvalues` | In multi-bit case: p-values of the statistical tests associated to each possible message |
| `payload` | In multi-bit case: extracted message (as an integer)  corresponding to the index with minimum p-value |


We provide examples of the output files in the `examples` folder as well as [logs](https://justpaste.it/21hj1).
They were obtained running the same command as shown above:
```cmd
python main_watermark.py --model_name guanaco-7b --prompt_type guanaco --prompt_path data/alpaca_data.json --nsamples 10 --batch_size 16 --method openai --method_detect openai --seeding hash --ngram 2 --scoring_method v2 --temperature 1.0 --payload 0 --payload_max 4 --output_dir examples/
```

### Evaluation only

We also provide a script to run the watermark detection algorithms on a given file of texts (for instance to run on vanilla data, or an already generated text).
You can use the script by running the `main_eval.py` file with the appropriate command-line arguments.

For instance runnning the following command should give the same results as the previous example:
```cmd
python main_eval.py \
    --json_path examples/results.jsonl --text_key result \
    --json_path_ori data/alpaca_data.json --text_key_ori output \
    --do_wmeval True --method openai --seeding hash --ngram 2 --scoring_method v2 \
    --payload 0 --payload_max 4 \
    --output_dir output/
```

This will save the results in the specified output directory. The output files are the same as for the watermark generation.

*Note*: Adding the arguments `--attack_name` and `--attack_param` allows to run the token substitution attack mentioned in the paper.



## Acknowledgements

Thanks to Vivien Chappelier for his help with the codebase, and Antoine Chaffin for the [Hugging Face Demo](https://huggingface.co/spaces/NohTow/LLaMav2_watermarking).

The watermarking methods are based on the works of [Kirchenbauer et al.](https://arxiv.org/abs/2301.10226) and of [Aaronson et al](https://www.scottaaronson.com/talks/watermark.ppt).

See also the following repositories:
- https://github.com/jwkirchenbauer/lm-watermarking
- https://github.com/facebookresearch/llama/


## License

three_bricks is CC-BY-NC licensed, as found in the [LICENSE](LICENSE) file.


## Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```
@article{fernandez2023three,
  title={Three Bricks to Consolidate Watermarks for Large Language Models},
  author={Fernandez, Pierre and Chaffin, Antoine and Tit, Karim and Chappelier, Vivien and Furon, Teddy},
  journal={preprint (under review)},
  year={2023}
}
```


