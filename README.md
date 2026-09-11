# DTRNet: Dynamic Token Routing Network

## Introduction

Transformers are widely used in Natural Language Processing (NLP), but the self-attention operation becomes expensive as the sequence length increases. Standard self-attention has a quadratic complexity of $\mathcal{O}(n^2)$ with respect to the sequence length.

During our analysis of **SmolLM-360M**, we observed that token representations in neighbouring inner layers were highly similar. The cosine similarity between consecutive layers was often close to **0.98**. This suggested that applying full attention to every token at every layer may perform unnecessary computation.

To reduce this cost, we propose **DTRNet (Dynamic Token Routing Network)**.

DTRNet uses a learned **per-token router** to decide which tokens should use the expensive attention operation. Around **10% of the tokens** are routed through full multi-head self-attention, while the remaining tokens use a cheaper path.

An important part of DTRNet is that **all tokens still pass through the shared MLP layer**. Therefore, tokens that skip attention are still updated at every layer. This is different from approaches such as **Mixture-of-Depths (MoD)** and **D-LLM**, where bypassed tokens can skip a larger part of the Transformer block.

The main goal of DTRNet is to reduce computation, latency, and KV-cache memory while maintaining good language-model performance.

---

## Implementation

DTRNet is implemented on top of **SmolLM-360M**.

The model contains a token router in selected inner Transformer layers. The router produces a score for each token and selects a small fraction of tokens for full attention.

The routing process can be summarized as:

```text
Input Tokens
     |
     v
 Token Router
     |
     +----------------------+
     |                      |
     v                      v
Selected Tokens       Bypassed Tokens
(~10%)                    (~90%)
     |                      |
     v                      v
Full Self-Attention    Cheap Linear Path
     |                      |
     +----------+-----------+
                |
                v
          Shared MLP
                |
                v
         Next Transformer Layer

The attention path is applied only to the selected tokens. The remaining tokens avoid the expensive quadratic attention computation.

However, the bypassed tokens are not simply left unchanged. They continue through the shared MLP layer, allowing their representations to evolve across layers.

---

## Key Features

* **Dynamic token routing**

  * A learned router selects tokens independently at each routed layer.

* **Sparse attention**

  * Only around 10% of tokens are sent through full attention.

* **MLP updates for all tokens**

  * Every token continues through the shared MLP layer.

* **Reduced attention computation**

  * Most tokens avoid the expensive quadratic attention operation.

* **Lower KV-cache memory**

  * Tokens that do not use attention do not need to create key/value states.

* **Long-sequence efficiency**

  * The reduction in attention computation becomes more useful as sequence length increases.

* **360M parameter model**

  * Experiments are performed using SmolLM-360M.

---

## Project Structure

```text
DTRNet/
│
├── model/
│   ├── dtrnet.py
│   ├── router.py
│   └── ...
│
├── training/
│   ├── train.py
│   └── ...
│
├── evaluation/
│   ├── evaluate.py
│   ├── latency.py
│   ├── flops.py
│   └── kv_cache.py
│
├── results/
│   ├── results_table.png
│   ├── flops_chart.png
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

---

# Experimental Setup

## Base Model

All experiments use:

* **Model:** SmolLM-360M
* **Parameters:** ~360M
* **Dataset:** FineWeb-Edu
* **Training tokens:** ~10B tokens
* **Maximum sequence length:** 1024
* **Precision:** bfloat16
* **Training:** Sequence packing

DTRNet routing is applied at the even-numbered inner layers:

```text
2, 4, 6, ..., 30
```

Approximately **10% of tokens** are routed to full attention at these layers.

---

## Training Configuration

| Setting                 | Value              |
| ----------------------- | ------------------ |
| Base Model              | SmolLM-360M        |
| Dataset                 | FineWeb-Edu        |
| Training Tokens         | ~10B               |
| Maximum Sequence Length | 1024               |
| Precision               | bfloat16           |
| Optimizer               | AdamW              |
| Learning Rate           | $3 \times 10^{-4}$ |
| Minimum Learning Rate   | $1 \times 10^{-5}$ |
| LR Schedule             | Cosine Decay       |
| Warmup                  | 0.1                |
| Weight Decay            | 0.01               |
| GPUs                    | 3                  |
| Per-device Batch Size   | 6                  |
| Gradient Accumulation   | 16                 |
| Global Batch Size       | 288                |
| Attention Routing Ratio | ~10%               |

---

# Evaluation

DTRNet is compared against:

* **Mixture-of-Depths (MoD)**
* **D-LLM**

The evaluation focuses on both **model quality** and **system efficiency**.

## Language Benchmarks

The model is evaluated using the `lm-evaluation-harness`.

The benchmark evaluation includes:

* WikiText
* LAMBADA
* ARC-Easy
* ARC-Challenge
* BoolQ
* HellaSwag
* PIQA
* MMLU
* Winogrande

The main comparison is performed at the **360M parameter scale**.

---

# Efficiency Evaluation

In addition to benchmark accuracy, we evaluate three important system-level metrics:

1. FLOPs
2. Generation latency
3. KV-cache memory

These measurements are important because reducing theoretical computation does not always translate directly into better practical inference performance.

---

## FLOPs vs Sequence Length

We estimate the FLOPs required by each routing method for different sequence lengths, including long sequences up to **20,000 tokens**.

The main observation is that DTRNet requires the lowest number of FLOPs across the tested sequence lengths.

At 20K tokens, the estimated FLOPs are approximately:

| Method |  FLOPs |
| ------ | -----: |
| DTRNet | ~16.5T |
| MoD    | ~19.5T |
| D-LLM  |   ~23T |

Therefore, at 20K tokens, DTRNet uses approximately:

* **15% fewer FLOPs than MoD**
* **28% fewer FLOPs than D-LLM**

The gap becomes larger as the sequence length increases.

This happens because attention has quadratic cost with sequence length. DTRNet sends only around 10% of tokens to attention, so the expensive attention computation is performed on a much smaller set of tokens.

![FLOPs vs Sequence Length](results/flops_chart.png)

---

## Generation Latency

We also measure generation latency to check whether the reduction in computation provides an actual speed improvement.

The measured results are:

| Method | Average Latency (ms) | Latency / Token (ms) |
| ------ | -------------------: | -------------------: |
| DTRNet |              1247.32 |                38.98 |
| MoD    |              1630.12 |                50.94 |
| D-LLM  |              5406.62 |               168.96 |

DTRNet achieves the lowest latency among the three routing methods.

This is important because FLOPs alone do not fully describe the real performance of a model. The latency results show that the reduced computation in DTRNet also translates into faster generation.

---

## KV-Cache Memory

During autoregressive generation, Transformer models store key/value states from previous tokens in the KV cache.

DTRNet reduces this memory usage because tokens that bypass attention do not need to create key/value states.

The measured KV-cache memory is:

| Sequence Length |   DTRNet |      MoD |    D-LLM |
| --------------- | -------: | -------: | -------: |
| 2K              |  44.3 MB |  75.3 MB |  80.0 MB |
| 4K              |  88.3 MB | 150.6 MB | 160.0 MB |
| 8K              | 177.7 MB | 301.2 MB | 320.0 MB |

At 8K sequence length, DTRNet reduces KV-cache memory by approximately:

* **41% compared with MoD**
* **44.5% compared with D-LLM**

The memory usage also grows approximately linearly with sequence length.

This lower KV-cache requirement is useful for long-context generation because it can allow longer inputs or larger batches within the same GPU memory budget.

![KV Cache Memory](results/kv_cache.png)

---

# Accuracy Results

DTRNet is evaluated against MoD and D-LLM at the **360M parameter scale**.

The average benchmark scores are:

| Method | Average Accuracy |
| ------ | ---------------: |
| DTRNet |        **44.21** |
| MoD    |            42.91 |
| D-LLM  |            41.34 |

DTRNet improves the average score by:

* **1.30 points over MoD**
* **2.87 points over D-LLM**

DTRNet also achieves the best results on several individual benchmarks, including ARC-Easy, PIQA, MMLU, and Winogrande.

For WikiText, the perplexity results are:

| Method | WikiText PPL |
| ------ | -----------: |
| DTRNet |        66.40 |
| MoD    |        68.75 |
| D-LLM  |        65.28 |

The results show that reducing attention computation does not require a large loss in model quality.

![Benchmark Results](results/results_table.png)

---

# Why DTRNet Works

The main difference between DTRNet and block-level skipping methods is that DTRNet does not completely stop updating bypassed tokens.

In DTRNet:

```text
Selected token
      |
      v
Full Attention
      |
      v
     MLP
      |
      v
Next Layer
```

while:

```text
Bypassed token
      |
      v
Cheap Path
      |
      v
     MLP
      |
      v
Next Layer
```

Therefore, even when a token does not require global attention, its representation can still change through the MLP.

This is useful because the high similarity between neighbouring layers suggests that many tokens may not need a completely new attention computation at every layer. Instead, they can receive cheaper updates while the model spends expensive attention computation only on tokens that need it.

This gives DTRNet a useful balance:

```text
Most Tokens
    |
    v
Cheap Update
    |
    v
Lower Compute


Important Tokens
    |
    v
Full Attention
    |
    v
Global Information
```

---

# Conclusion

DTRNet is a dynamic token routing approach designed to reduce the cost of Transformer self-attention.

Instead of applying full attention to every token, DTRNet routes only around **10% of tokens** to attention while the remaining tokens use a cheaper path. Importantly, all tokens still pass through the shared MLP layer, so bypassed tokens continue to receive updates.

The experiments on SmolLM-360M show that DTRNet provides a strong balance between accuracy and efficiency. It achieves the highest average benchmark accuracy among the compared routing methods while also achieving the lowest measured latency, FLOPs, and KV-cache memory.

The efficiency gains become more important for longer sequences because the cost of self-attention grows rapidly with sequence length. The lower KV-cache usage is also useful for long-context inference and memory-constrained deployment.

Overall, the results suggest that **attention does not need to be applied equally to every token at every layer**. Dynamically selecting the tokens that need expensive attention can reduce computation while maintaining strong model performance.

---

# Future Work

Several directions can be explored in future work.

### Larger Models

DTRNet can be tested on larger language models to study whether the same routing behaviour continues to work as model size increases.

### Longer Training

Longer training schedules can be used to study whether the model can further improve its accuracy after the initial training stage.

### Adaptive Routing

Instead of keeping the routing ratio fixed at around 10%, the router could dynamically choose different numbers of tokens depending on the layer or input.

For example:

```text
Early Layer       → More Attention
Middle Layer      → Less Attention
Later Layer       → More Attention
```

This could potentially provide a better compute-quality trade-off.

### Long-Context Evaluation

DTRNet can be tested on more long-context tasks such as:

* Long-document understanding
* Document summarization
* Code analysis
* Long-context question answering
* Long-context generation

### Router Analysis

Another useful direction is to study which tokens are selected by the router and why they receive attention. This could provide a better understanding of how the model decides which tokens need global information.

### Combination with Other Efficiency Methods

DTRNet can also be combined with techniques such as quantization and optimized decoding methods to further reduce inference cost.

---

# Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd DTRNet
pip install -r requirements.txt
```

---

# Training

The model can be trained using the provided training script:

```bash
python training/train.py
```

The training configuration can be adjusted for:

* Learning rate
* Batch size
* Gradient accumulation
* Number of training tokens
* Sequence length
* Routing ratio
* Routing layers

---

# Evaluation

Benchmark evaluation can be run using:

```bash
python evaluation/evaluate.py
```

Efficiency measurements can be run using the corresponding evaluation scripts:

```bash
python evaluation/latency.py
python evaluation/flops.py
python evaluation/kv_cache.py
```

---

# Results Summary

| Metric               |       DTRNet |      MoD |    D-LLM |
| -------------------- | -----------: | -------: | -------: |
| Average Accuracy     |    **44.21** |    42.91 |    41.34 |
| Avg. Latency (ms)    |  **1247.32** |  1630.12 |  5406.62 |
| Latency / Token (ms) |    **38.98** |    50.94 |   168.96 |
| KV Cache @ 8K        | **177.7 MB** | 301.2 MB | 320.0 MB |
| FLOPs @ 20K          |   **~16.5T** |   ~19.5T |     ~23T |
| Attention Routing    |     **~10%** |        — |     ~84% |

---

# Acknowledgements

This project was developed as part of our work in SML(Systems for Machine Learning) course project.

We use **SmolLM-360M** as the base language model and **FineWeb-Edu** for training. Benchmark evaluation is performed using the `lm-evaluation-harness`.

```
```
