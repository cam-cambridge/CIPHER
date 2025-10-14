# CIPHER : Control and Interpretation of Production via Hybrid Expertise and Reasoning 

<p align="center">
  <img src="assets/teaser.png" alt="CIPHER controlling a 3-D printer" width="100%">
  <br>
  <em>Figure&nbsp;1 – CIPHER translates textual/visual prompts into printer commands, through physics- and geometry-informed reasoning.</em>
</p>

### Autonomous design and printing

<p align="center">
  <img src="assets/geometries.png" alt="AI desings and prints its own parts" width="100%">
  <br>
  <em>Figure&nbsp;2 – Send pics of your AI-generated printed parts at {cm2161@cam.ac.uk} to be featured in our project page!</em>
</p>

<hr style="border: 2px solid gray;"></hr>

### Codebase structure

```
CIPHER/
├── requirements.txt          # install Python dependencies
├── src/                    
│   ├── config.py             # Configuration and hyperparameters
│   ├── model.py              # Main VLA model implementation
│   ├── vexpert.py            # Vision expert for process monitoring
│   ├── train.py              # Training loops and callbacks
│   ├── main.py               # Training loops and callbacks
│   └── utils
│       ├── data_utils.py   
│       ├── test_utils.py   
│       └── utils.py
├── scripts/
├── prompts/
└── assets/

```

<hr style="border: 2px solid gray;"></hr>

## Setup environment
### Requirements and setup
- cuda>=11
- torch>=1.7
- Python >= 3.11
```bash
git clone git@github.com:cam-cambridge/CIPHER.git
cd CIPHER
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_********************************
```
where ******************************** is your HF key (see https://huggingface.co/docs/hub/en/security-tokens)

## Train

```bash
bash scripts/train.sh
```
The pre-trained [microsoft/ResNet-50](https://huggingface.co/microsoft/resnet-50) model and the pre-trained [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) will be fetched from Hugging Face. The train dataset (subset) will be fetched from [cemag/tl-caxton](https://huggingface.co/datasets/cemag/tl-caxton).

We train on 4 × NVIDIA A100 80GB. LoRA models can be trained with significantly fewer resources.

Explicit Notes on Model Licensing & Commercial Use: While all code in this repository is released under an MIT License, our pretrained models may inherit restrictions from the underlying base models we use. Specifically, CIPHER is derived from Llama-3.2, and as such are subject to the Llama Community License.

## Inference / Tests
We have prepared test scripts for the experiments as seen in the paper.
Available scripts:
| Script | Description | Inputs |
|--------|-------------|--------|
| `scripts/ask.sh` | Ask a single question to the model | `--model_path`<br>`--question`<br>`--results_path` |
| `scripts/test_flowrate_predictions.sh` | Test flowrate predictions on test dataset | `--test_samples`<br>`--batch_size`<br>`--model_path`<br>`--data_path`<br>`--results_path` |
| `scripts/test_vanilla_control.sh` | Test vanilla control performance | `--model_path`<br>`--num_questions`<br>`--prompt_path`<br>`--results_path` |
| `scripts/test_domain_expertise.sh` | Test domain expertise with/without RAG | `--model_path`<br>`--questions_path`<br>`--rag`<br>`--results_path`<br>`--context` |
| `scripts/test_overfit.sh` | Test models catastrophic forgetting on SQUAD (language) and Flickr30 (image)datasets | `--test_samples`<br>`--model_path`<br>`--results_path` |

Each script has a help menu accessible via `-h` or `--help` flag.



<hr style="border: 2px solid gray;"></hr>

#### Citation

⭐ If you find our code or models useful in your work, please cite our paper:

Christos Margadji & Sebastian W. Pattinson (2025). *Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing*. [arXiv:2506.08462](https://arxiv.org/abs/2506.08462)
```bash
@article{MargadjiPattinson2025HybridReasoning,
  title   = {Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing},
  author  = {Margadji, Christos and Pattinson, Sebastian W.},
  year    = {2025},
  note    = {arXiv:2506.08462},
  url     = {https://arxiv.org/abs/2506.08462}
}
```