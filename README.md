# DTRNet: Dynamic Token Routing Network

DTRNet is a compute-efficient Transformer architecture that reduces the cost of self-attention using **dynamic token routing**.

Our analysis of **SmolLM-360M** showed that neighbouring inner layers often produce highly similar token representations, with cosine similarity close to **0.98**. This motivated us to avoid applying expensive attention to every token at every layer.

DTRNet uses a learned **per-token router** to send only around **10% of tokens** through full self-attention. The remaining tokens use a cheaper path. Importantly, **all tokens still pass through the shared MLP**, so bypassed tokens continue to receive updates.

## Key Idea

```text
                    Token Router
                         |
             +-----------+-----------+
             |                       |
        ~10% Tokens             ~90% Tokens
             |                       |
      Full Attention             Cheap Path
             |                       |
             +-----------+-----------+
                         |
                    Shared MLP
                         |
                    Next Layer
````

This reduces the expensive attention computation while maintaining continuous updates for all tokens.

## Repository Structure

```text
SML-Project/
├── configs/              # Training configurations
├── experiments/          # Experiment configurations
├── src/                  # Model and training code
├── scripts/              # Utility scripts
├── results/              # Evaluation results and plots
├── tests/                # Tests
├── lm-evaluation-harness/
├── train_pt.sh           # Training script
├── requirements.txt
└── pyproject.toml
```

## Experimental Setup

* **Base Model:** SmolLM-360M
* **Dataset:** FineWeb-Edu
* **Training:** ~10B tokens
* **Maximum Sequence Length:** 1024
* **Precision:** bfloat16
* **Optimizer:** AdamW
* **GPUs:** 3
* **Per-device Batch Size:** 6
* **Gradient Accumulation:** 16
* **Global Batch Size:** 288
* **Routing Ratio:** ~10%
* **Routing Layers:** 2, 4, 6, ..., 30

## Training

Clone the repository and install the dependencies:

```bash
git clone https://github.com/abhigna333/SML-Project.git
cd SML-Project
pip install -r requirements.txt
```

Run training using:

```bash
bash train_pt.sh
```

The main experiment configuration is:

```text
experiments/smollm_360M_fineweb_15B_PT.yaml
```

## Evaluation

DTRNet is compared with **Mixture-of-Depths (MoD)** and **D-LLM**.

We evaluate:

* Language-model benchmark accuracy
* FLOPs
* Generation latency
* KV-cache memory

Benchmark evaluation uses the included `lm-evaluation-harness`.

## Results

### Accuracy

| Method     | Average Accuracy |
| ---------- | ---------------: |
| **DTRNet** |        **44.21** |
| MoD        |            42.91 |
| D-LLM      |            41.34 |

DTRNet improves the average score by **1.30 points over MoD** and **2.87 points over D-LLM**.

### Efficiency

| Metric               |       DTRNet |      MoD |    D-LLM |
| -------------------- | -----------: | -------: | -------: |
| Avg. Latency (ms)    |  **1247.32** |  1630.12 |  5406.62 |
| Latency / Token (ms) |    **38.98** |    50.94 |   168.96 |
| KV Cache @ 8K        | **177.7 MB** | 301.2 MB | 320.0 MB |
| FLOPs @ 20K          |   **~16.5T** |   ~19.5T |     ~23T |

At 20K tokens, DTRNet uses approximately **15% fewer FLOPs than MoD** and **28% fewer than D-LLM**.

At 8K tokens, DTRNet uses approximately **41% less KV-cache memory than MoD** and **44.5% less than D-LLM**.

## Results

The plots and evaluation outputs are available in the `results/` directory.

## Future Work

Future work includes:

* Testing DTRNet on larger models
* Learning adaptive routing ratios across layers
* Evaluating longer-context tasks
* Further analysis of router decisions
* Combining DTRNet with quantization and other inference optimizations

## Acknowledgements

This project was developed as part of an **SML course project**. We use **SmolLM-360M**, **FineWeb-Edu**, and the **lm-evaluation-harness**.


