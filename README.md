<p align="center">
<h1 align="center">
  <img src="assets/title_github.png" width="800px" alt="A Systematic Survey of Optimization Methods">
</h1>

<div align="center">

[![Torch Hub Support](https://img.shields.io/badge/torch_hub-gray?style=for-the-badge&logo=pytorch)](#torch-hub) [![Github](https://img.shields.io/badge/optim--survey-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/JZhangTon/optim_survey)

</div>

This is the official implementation for our paper:  
[Tong Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=WsEUmpwAAAAJ&view_op=list_works&gmla=AF9nlQsB3CdUdRLrWHi9n_CwlbjpAQF4s01SgZMA212-nS3JzQjGGYmH_SRMgvuu1AHgiXAIw0h81427vtjl_3_IeQwZ9GUm0nlwcPbu9Jc)
.
[Jiangning Zhang](https://zhangzjn.github.io)\*

\*: Equal contribution.

**🔥Add Your Paper in our Survey!!!!!**

[-] You are welcome to give us an issue or PR for your open vocabulary learning work !!!!!

[-] Note that: Due to the huge paper in Arxiv, we are sorry to cover all in our survey. You can directly present a PR into this repo and we will record it for next version update of our survey.

**🔥New**
 - **[2026.03.04]** We update GitHub to record the available paper by the end of 2026/3/4.

## 🔨Installation
OptimSurvey requires a reasonably recent version of [torch](https://pytorch.org/get-started/locally/).
After that, You can install our optimizer library via [pip](https://github.com/JZhangTon/awesome-optimizer.git):
```bash
pip install optim_survey
```
### 🔨Installing from Source
If you want to reproduce the benchmarks or develop new optimizers based on our template:
```bash
git clone https://github.com/JZhangTon/awesome-optimizer.git
cd awesome-optimizer
```
## 📌Introduction
This survey presents the first detailed survey on modern deep learning optimization methods, including adaptive methods (e.g., AdamW), sign-based optimization (e.g., Lion), memory-efficient optimizers for Large Language Models, and specific optimization methods for FL and DL.

<img src="assets/drawing_main.png" width.="1000px">

## 🗂️Summary of Contents

- [📌Introduction](#introduction)
- [🗂️Summary of Contents](#summary-of-contents)
- [📊Methods: A Survey](#methods-a-survey)
  - [📄Taxonomy of Optimization Methods](#taxonomy-of-optimization-methods)
- [⚙️Usage](#usage)
- [📈Benchmarking](#benchmarking)
- [🔗Citation](#citation)
- [📫Contact](#contact)
## 📊Methods: A Survey
**🗝️Keywords**

- `Gradient Descent(GD).`: The Gradient Descent algorithm is an iterative optimization process applied to a differentiable loss function.
- `Momentum.`: Alongside the standard gradient update, Momentum includes a velocity vector that accumulates past gradients to smooth out the optimization trajectory.
- `Adaptive Methods.`: The Adaptive Methods are a collection of algorithms that adjust the learning rate based on parameter granularity or structure.
### 📄Taxonomy of Optimization Methods
|Optimizer Taxonomy|Sub-methods|Fine-grained Methods|idx|Abbreviation|Venue|Year|Paper Title|Project|
|---|---|---|:-:|:-:|:-:|:-:|---|---|
|First-Order Methods|Learning Rate Scheduling|Stability-Aware Adaptive Scheduling|1|Multistage SGDM|NeurIPS|2020|An Improved Analysis of Stochastic Gradient Descent with Momentum|[Link](https://proceedings.neurips.cc/paper/2020/hash/d3f5d4de09ea19461dab00590df91e4f-Abstract.html)|
|First-Order Methods|Learning Rate Scheduling|Stability-Aware Adaptive Scheduling|2||ICML|2023|SGD with Large Step Sizes Learns Sparse Features|[Link](https://arxiv.org/abs/2210.05337)|
|First-Order Methods|Gradient Normalization & Clipping|Element-Wise Gradient Scaling|3|pbSGD|IJCAI|2020|pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non-Convex Optimization|[Link](https://www.ijcai.org/proceedings/2020/451)|
||||4||NeurIPS|2024|Stochastic Taylor Derivative Estimator: Efficient amortization for arbitrary differential operators|[Link](https://openreview.net/forum?id=J2wI2rCG2u)|
|First-Order Methods|Hybrid Methods|Gradient Filtering Hybrid|5|ADASS|arXiv|2019|ADASS: Adaptive Sample Selection for Training Acceleration|[Link](https://arxiv.org/abs/1906.04819)|
|First-Order Methods|Adaptive Learning Rate Methods； Hybrid Methods|Hybrid Adaptive Strategy； SGD-Adam Hybrid|6|AdaSGD|arXiv|2020|AdaSGD: Bridging the gap between SGD and Adam|[Link](https://arxiv.org/abs/2006.16541)|
|First-Order Methods|Gradient Normalization & Clipping|Basic Fixed Gradient Clipping|7|clipped-SGD|NeurIPS|2020|Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping|[Link](https://arxiv.org/abs/2005.10785)|
|First-Order Methods|Hybrid Methods|Projection Gradient Hybrid|8|Cayley SGD|ICLR|2020|EFFICIENT RIEMANNIAN OPTIMIZATION ON THE STIEFEL MANIFOLD VIA THE CAYLEY TRANSFORM|[Link](https://arxiv.org/abs/2002.01113)|
|First-Order Methods|Learning Rate Scheduling|Scheduler-Free Adaptation|9|ADAS|arXiv|2020|ADAS: ADAPTIVE SCHEDULING OF STOCHASTIC GRADIENTS|[Link](https://arxiv.org/abs/2006.06587)|
|First-Order Methods|Momentum-Enhanced SGD|Scheduled Momentum Reset|10|SRSGD|SIAM Journal on Imaging Sciences|2022|Scheduled Restart Momentum for Accelerated Stochastic Gradient Descent|[Link](https://arxiv.org/abs/2002.10583)|
|First-Order Methods|Learning Rate Scheduling|Scheduler-Free Adaptation|11|SGD-G2|ICPR|2021|Stochastic Runge-Kutta methods and adaptive SGD-G2 stochastic gradient descent|[Link](https://arxiv.org/abs/2002.09304)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|12|BADM|arXiv|2024|BADM: Batch ADMM for Deep Learning|[Link](https://arxiv.org/abs/2407.01640)|
|First-Order Methods； Memory-Efficient Optimization Methods|Learning Rate Scheduling； Stateless Optimization Methods|Initial Learning Rate Scaling； Parameter Characteristic-Driven Updates|13|SGD-SaI|arXiv|2024|No More Adam: Learning Rate Scaling at Initialization is All You Need|[Link](https://arxiv.org/abs/2412.11768)|
|First-Order Methods|Momentum-Enhanced SGD|Double-momentum mechanism|14|μ²-SGD|ICLR|2025|DO STOCHASTIC, FEEL NOISELESS: STABLE STOCHASTIC OPTIMIZATION VIA A DOUBLE MOMENTUM MECHANISM|[Link](https://arxiv.org/abs/2304.04172)|
|First-Order Methods|Learning Rate Scheduling|Scheduler-Free Adaptation|15|AutoSGD|arXiv|2025|AutoSGD: Automatic Learning Rate Selection for Stochastic Gradient Descent|[Link](https://arxiv.org/abs/2505.21651)|
|Zeroth-Order Methods|Memory-efficient Methods|Inference-Level Memory Zeroth-Order|16|MeZO|NeurIPS|2023|Fine-Tuning Language Models with Just Forward Passes|[Link](https://arxiv.org/abs/2305.17333)|
|First-Order Methods； Memory-Efficient Optimization Methods|Towards LLM Traning； Memory-Efficient Fine-Tuning for Large Models|Real-Time Computation； Staless Fine-Tuning|17|LOMO|ACL|2024|Full Parameter Fine-tuning for Large Language Models with Limited Resources|[Link](https://arxiv.org/abs/2306.09782)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|18|AdaLOMO|arXiv|2023|AdaLomo: Low-memory Optimization with Adaptive Learning Rate|[Link](https://arxiv.org/abs/2310.10195)|
|Zeroth-Order Methods|Variance Reduction|Snapshot Variance Reduction|19|MeZO-SVRG|ICLR|2024|Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models|[Link](https://arxiv.org/abs/2404.08080)|
|Zeroth-Order Methods|Perturbation Optimization； Memory-efficient Methods|Paired Perturbation Sampling； Inference-Level Memory Zeroth-Order|20|ZO-AdaMU|AAAI|2024|ZO-AdaMU Optimizer: Adapting Perturbation by the Momentum and Uncertainty in Zeroth-order Optimization|[Link](https://arxiv.org/abs/2312.15184)|
|First-Order Methods|Hybrid Methods； Towards LLM Traning|Multi-Objective Hybrid； Real-Time Computation； Block-Wise Computation|21|BAdam|NeurIPS|2024|BAdam: A Memory Efficient Full Parameter Optimization Method for Large Language Models|[Link](https://arxiv.org/abs/2404.02827)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Evolutionary Strategies&Meta-Adaptive Learning|22|MADA|ICML|2024|MADA: Meta-Adaptive Optimizers through hyper-gradient Descent|[Link](https://arxiv.org/abs/2401.08893)|
|Zeroth-Order Methods； Memory-Efficient Optimization Methods|Perturbation Optimization； Memory-efficient Methods； Memory-Efficient Fine-Tuning for Large Models|Sparse Perturbation； Sparse Parameter Zeroth-Order； Selective Parameter Fine-Tuning|23|Sparse MeZO|arXiv|2024|Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning|[Link](https://arxiv.org/abs/2402.15751)|
|Zeroth-Order Methods|Memory-efficient Methods|Low-Rank Zeroth-Order Finetuning|24|LOZO|ICLR|2025|Enhancing zeroth-order fine-tuning for language models with low-rank structures|[Link](https://arxiv.org/abs/2410.07698)|
|Memory-Efficient Optimization Methods|Low-Rank Gradient Storage|Dynamic Gradient Rank|25|A-GNB|arXiv|2024|HELENE: HESSIAN LAYER-WISE CLIPPING AND GRADIENT ANNEALING FOR ACCELERATING FINETUNING LLM WITH ZEROTH-ORDER OPTIMIZATION|[Link](https://arxiv.org/abs/2411.10696)|
|Memory-Efficient Optimization Methods|Low-Rank Methods|Projection\&Adjustment|26|AdaRankGrad|ICLR|2025|Adarankgrad: Adaptive gradient-rank and moments for memory-efficient llms training and fine-tuning|[Link](https://arxiv.org/abs/2410.17881)|
|Zeroth-Order Methods|Adaptive Methods|Momentum-based Adaptive； Low-Rank Adaptive|27|LORENZA|arXiv|2025|LORENZA: Enhancing Generalization in Low-Rank Gradient LLM Training and Fine-Tuning via Efficient Zeroth-Order Adaptive SAM Optimization|[Link](https://arxiv.org/abs/2502.19571)|
|First-Order Methods； Privacy-Preserving Optimization Methods|Gradient Normalization & Clipping； Privacy-Aware Gradient Clipping|Layer-Wise Gradient Normalization； Dynamic Gradient Clipping； Adaptive Clipping|28|Stable-SPAM|ICLR|2025|Stable-SPAM: How to Train in 4-Bit More Stably than 16-Bit Adam|[Link](https://arxiv.org/abs/2502.17055)|
|Zeroth-Order Methods； Memory-Efficient Optimization Methods|Memory-efficient Methods； Low-Rank Methods|Quantized Zeroth-Order Finetuning； Low-Rank & Quantization|29|QuZO|arXiv|2025|QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models|[Link](https://arxiv.org/abs/2502.12346)|
|First-Order Methods； Other Specialized Optimization Methods|Gradient Normalization & Clipping； Robust Optimization|Noise-Robust Normalization； Dynamic Gradient Clipping； Noise-Robust Gradients|30|AdaGC|arXiv|2025|AdaGC: Improving Training Stability for Large Language Model Pretraining|[Link](https://arxiv.org/abs/2502.11034)|
|Zeroth-Order Methods|Memory-efficient Methods|Sparse Parameter Zeroth-Order|31|MaZO|arXiv|2025|MaZO: Masked Zeroth-Order Optimization for Multi-Task Fine-Tuning of Large Language Models|[Link](https://arxiv.org/abs/2502.11513)|
|Zeroth-Order Methods|Memory-efficient Methods|Inference-Level Memory Zeroth-Order|32|ZO2|arXiv|2025|ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory|[Link](https://arxiv.org/abs/2503.12668)|
|First-Order Methods|Adaptive Learning Rate Methods|Stateless Adaptation|33|AdamS|arXiv|2025|AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training|[Link](https://arxiv.org/abs/2505.16363)|
|Memory-Efficient Optimization Methods|Low-Rank Gradient Storage|Gradient Low-Rank Projection|34|SUMO|arXiv|2025|SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training|[Link](https://arxiv.org/abs/2505.24749)|
|Zeroth-Order Methods|Memory-efficient Methods|Inference-Level Memory Zeroth-Order|35|KerZOO|arXiv|2025|KerZOO: Kernel Function Informed Zeroth-Order Optimization for Accurate and Accelerated LLM Fine-Tuning|[Link](https://arxiv.org/abs/2505.18886)|
|Zeroth-Order Methods|Memory-efficient Methods|Quantized Zeroth-Order Finetuning|36|QZO|arXiv|2025|Fine-tuning Quantized Neural Networks with Zeroth-order Optimization|[Link](https://arxiv.org/abs/2505.13430)|
|Zeroth-Order Methods|Variance Reduction|Structured Variance Control|37|FZOO|arXiv|2025|FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed|[Link](https://arxiv.org/abs/2506.09034)|
|Zeroth-Order Methods|Memory-efficient Methods|Low-Rank Zeroth-Order Finetuning|38|TeZO|arXiv|2025|TeZO: Empowering the Low-Rankness on the Temporal Dimension in the Zeroth-Order Optimization for Fine-tuning LLMs|[Link](https://arxiv.org/abs/2501.19057)|
|First-Order Methods|Loss Landscape Optimization|Momentum Landscape Adaptation|39|SCSAdamW|arXiv|2025|Beyond First-Order: Training LLMs with Stochastic Conjugate Subgradients and AdamW|[Link](https://arxiv.org/abs/2507.01241)|
|First-Order Methods|Momentum-Enhanced SGD|Accelerated Momentum|40|adaNPAG|arXiv|2025|Boosting Accelerated Proximal Gradient Method with Adaptive Sampling for Stochastic Composite Optimization *|[Link](https://arxiv.org/abs/2507.18277)|
|Memory-Efficient Optimization Methods|Low-Rank Gradient Storage|Gradient Low-Rank Projection|41|GWT|arXiv|2025|Wavelet Meets Adam: Compressing Gradients for Memory-Efficient Training|[Link](https://arxiv.org/abs/2501.07237)|
|Zeroth-Order Methods|Adaptive Methods|Projection-based Adaptive|42|DiZO|arXiv|2025|Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning|[Link](https://arxiv.org/abs/2502.03304)|
|First-Order Methods|Adaptive Learning Rate Methods； Towards LLM Traning|Hybrid Adaptive Strategy； Gradient Projection Mechanism|43|apollo|MLSys|2025|APOLLO:SGD-LIKE MEMORY, ADAMW-LEVEL PERFORMANCE|[Link](https://arxiv.org/abs/2412.05270)|
|First-Order Methods|Adaptive Learning Rate Methods|Hybrid Adaptive Strategy|44|SoftSignSGD|arXiv|2025|SoftSignSGD(S3): An Enhanced Optimizer for Practical DNN Training and Loss Spikes Minimization Beyond Adam|[Link](https://arxiv.org/abs/2507.06464)|
|Memory-Efficient Optimization Methods|Low-Memory Optimizer Design|Structural Redesign|45|Adam-mini|ICLR|2025|ADAM-MINI: USE FEWER LEARNING RATES TO GAIN MORE|[Link](https://arxiv.org/abs/2406.16793)|
|First-Order Methods； Memory-Efficient Optimization Methods|Gradient Normalization & Clipping； Optimizer State Compression|Element-Wise Gradient Scaling； Spike-Aware Gradient Clipping； Sparse State Compression|46|SPAM|ICLR|2025|SPAM: SPIKE-AWARE ADAM WITH MOMENTUM RESET FOR STABLE LLM TRAINING|[Link](https://arxiv.org/abs/2501.06842)|
|Memory-Efficient Optimization Methods|Optimizer State Compression|Sparse State Compression|47|MICROADAM|NeurIPS|2024|MICROADAM: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence|[Link](https://arxiv.org/abs/2405.15593)|
|First-Order Methods|Adaptive Learning Rate Methods|Layer-Wise Adaptation|48|Coupled Adam|ACL|2025|Better Embeddings with Coupled Adam|[Link](https://arxiv.org/abs/2502.08441)|
|First-Order Methods|Towards LLM Traning|Gradient Preconditioning Mechanism|49|SWAN|ICML|2025|SWAN: SGD WITH NORMALIZATION AND WHITENING ENABLES STATELESS LLM TRAINING|[Link](https://arxiv.org/abs/2412.13148)|
|Zeroth-Order Methods； Memory-Efficient Optimization Methods|Perturbation Optimization； Memory-efficient Methods； Memory-Efficient Fine-Tuning for Large Models|Sparse Perturbation； Sparse Parameter Zeroth-Order； Selective Parameter Fine-Tuning|50|LeZO|arXiv|2024|SIMULTANEOUS COMPUTATION AND MEMORY EFFICIENT ZEROTH-ORDER OPTIMIZER FOR FINE-TUNING LARGE LANGUAGE MODELS|[Link](https://arxiv.org/abs/2410.09823)|
|Zeroth-Order Methods|Memory-efficient Methods|Low-Rank Zeroth-Order Finetuning|51|SuZero|arXiv|2024|Zeroth-Order Fine-Tuning of LLMs in Random Subspaces|[Link](https://arxiv.org/abs/2410.08989)|
|Zeroth-Order Methods|Zeroth-First Order Hybrid|Weighted Hybrid|52|Addax|ICLR|2025|Addax: Utilizing Zeroth-Order Gradients to Improve Memory Efficiency and Performance of SGD for Fine-Tuning Language Models|[Link](https://arxiv.org/abs/2410.06441)|
|First-Order Methods|Hybrid Methods|Projection Gradient Hybrid|53|LDAdam|ICLR|2025|LDADAM: ADAPTIVE OPTIMIZATION FROM LOWDIMENSIONAL GRADIENT STATISTICS|[Link](https://arxiv.org/abs/2410.16103)|
|First-Order Methods|Hybrid Methods|Projection Gradient Hybrid|54|Adadiag|arXiv|2025|Improving Adaptive Moment Optimization viaPreconditioner Diagonalization|[Link](https://arxiv.org/abs/2502.07488)|
|Privacy-Preserving Optimization Methods|Privacy-Utility Tradeoff|Post-Processing Optimization|55|DP-LSSGD|PMLR|2020|DP-LSSGD: A Stochastic Optimization Method to Lift the Utility in Privacy-Preserving ERM|[Link](https://arxiv.org/abs/1906.12056)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|56|DP-SGD-JL|NeurIPS|2021|Fast and Memory Efficient Differentially Private-SGD via JL Projections|[Link](https://arxiv.org/abs/2102.03013)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|57|DPIS|CCS|2022|DPIS: An Enhanced Mechanism for Differentially Private SGD with Importance Sampling|[Link](https://arxiv.org/abs/2210.09634)|
|Zeroth-Order Methods； Privacy-Preserving Optimization Methods|Distributed Zero-Order Optimization； Differential Privacy Optimization|Privacy-Preserving Zeroth-Order； DP-SGD Variants； Dynamic Noise Scheduling； Privacy-Utility Balance|58|TOP-DP|IEEE Trans|2021|Topology-aware Differential Privacy for Decentralized Image Classification|[Link](https://arxiv.org/abs/2006.07817)|
|First-Order Methods|Gradient Normalization & Clipping|Basic Fixed Gradient Clipping|59|DP-SGD|arXiv|2022|Normalized/Clipped SGD with Perturbation for Differentially Private Non-Convex Optimization|[Link](https://arxiv.org/abs/2206.13033)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization； Gradient Noise Injection|DP-SGD Variants； Noise-Robust Optimization|60|DP-Adam|ICLR|2023|DP-ADAM: CORRECTING DP BIAS IN ADAM’S SECOND MOMENT ESTIMATION|[Link](https://arxiv.org/abs/2304.11208)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|61|ANSGD|arXiv|2023|Learning across Data Owners with Joint Differential Privacy|[Link](https://arxiv.org/abs/2305.15723)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|62|DP-AdamBC|AAAI|2024|DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)|[Link](https://arxiv.org/abs/2312.14334)|
|First-Order Methods|Gradient Normalization & Clipping|DP-enhanced Gradient Clipping|63|Dice-SGD|ICLR|2024|DIFFERENTIALLY PRIVATE SGD WITHOUT CLIPPING BIAS: AN ERROR-FEEDBACK APPROACH|[Link](https://arxiv.org/abs/2311.14632)|
|Privacy-Preserving Optimization Methods|Privacy-Aware Gradient Clipping|Global Gradient Clipping|64|AClipped-dpSGD|Mach. Learn.|2024|Efficient Private SCO for Heavy-Tailed Data via Averaged Clipping|[Link](https://arxiv.org/abs/2206.13011)|
|Privacy-Preserving Optimization Methods|Gradient Noise Injection|Noise-Robust Optimization|65|GeoDP-SGD|ICDE|2025|Analyzing and Optimizing Perturbation of DP-SGD Geometrically|[Link](https://arxiv.org/abs/2504.05618)|
|Privacy-Preserving Optimization Methods|Privacy-Aware Gradient Clipping|Global Gradient Clipping|66|Logit-DP|ICLR|2025|DIFFERENTIALLY PRIVATE OPTIMIZATION FOR NONDECOMPOSABLE OBJECTIVE FUNCTIONS|[Link](https://arxiv.org/abs/2310.03104)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|67|DOPPLER|NeurIPS|2025|DOPPLER: Differentially Private Optimizers with Low-pass Filter for Privacy Noise Reduction|[Link](https://arxiv.org/abs/2408.13460)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization； Gradient Noise Injection； Privacy-Utility Tradeoff； Privacy-Aware Gradient Clipping|Dynamic Noise Scheduling； Dynamic Clipping Threshold； Adaptive Clipping|68|DC-SGD|TIFS|2025|DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation|[Link](https://arxiv.org/abs/2503.22988)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|69|SPARTA|KDD|2025|SPARTA: An Optimization Framework for Differentially Private Sparse Fine-Tuning|[Link](https://arxiv.org/abs/2503.12822)|
|Distributed Optimization Methods； Privacy-Preserving Optimization Methods|Decentralized Communication； Differential Privacy Optimization|Privacy-Preserving Decentralization； Privacy-Utility Balance|70|Interleaved-ShuffleG|arXiv|2025|The Cost of Shuffling in Private Gradient Based Optimization|[Link](https://arxiv.org/abs/2502.03652)|
||||71||NeurIPS|2022|Training compute-optimal large language models|[Link](https://doi.org/10.48550/arXiv.2203.15556)|
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|72|RaCO-DP|arXiv|2025|Private Rate-Constrained Optimization with Applications to Fair Learning|[Link](https://arxiv.org/abs/2505.22703)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Low-Rank Gradient Compression|73|DOME|arXiv|2025|Communication Efficient, Differentially Private Distributed Optimization using Correlation-Aware Sketching|[Link](https://arxiv.org/abs/2507.03545)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Low-Rank Gradient Compression|74|PowerSGD|NeurIPS|2019|PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization|[Link](https://arxiv.org/abs/1905.13727)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Quantization Compression|75|signProx|ICASSP|2019|signProx: One-Bit Proximal Algorithm for Nonconvex Stochastic Optimization|[Link](https://arxiv.org/abs/1807.08023)|
|Distributed Optimization Methods|Local Update Strategies|Local SGD|76|DLCP|arXiv|2020|Domain-specific Communication Optimization for Distributed DNN Training|[Link](https://arxiv.org/abs/2008.08445)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Quantization Compression|77|DEED-GD|arXiv|2020|DEED: A General Quantization Scheme for Communication Efficiency in Bits|[Link](https://arxiv.org/abs/2006.11401)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Adaptive Compression Level|78|LAGS-SGD|ECAI|2020|Layer-wise Adaptive Gradient Sparsification for Distributed Deep Learning with Convergence Guarantees|[Link](https://arxiv.org/abs/1911.08727)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|79|FedAC|NeurIPS|2020|Federated Accelerated Stochastic Gradient Descent|[Link](https://arxiv.org/abs/2006.08950)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Sparsification Compression|80|rTop-k|JSAIT|2020|rTop-k: A Statistical Estimation Approach to Distributed SGD|[Link](https://arxiv.org/abs/2005.10761)|
|Distributed Optimization Methods|Distributed Hybrid Optimization|Compression & Local Updates|81|Qsparse-local-SGD|JSAIT|2020|Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations|[Link](https://arxiv.org/abs/1906.02367)|
|Distributed Optimization Methods|Local Update Strategies|Local SGD|82|SLOWMO|ICLR|2020|SLOWMO: IMPROVING COMMUNICATION-EFFICIENT DISTRIBUTED SGD WITH SLOW MOMENTUM|[Link](https://arxiv.org/abs/1910.00643)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Compression Error Compensation|83|APMSqueeze|arXiv|2020|APMSqueeze: A Communication Efficient Adam-Preconditioned Momentum SGD Algorithm|[Link](https://arxiv.org/abs/2008.11343)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Quantization Compression|84|1-bit Adam|ICML|2021|1-bit Adam: Communication Efficient Large-Scale Training with Adam’s Convergence Speed|[Link](https://arxiv.org/abs/2102.02888)|
|Distributed Optimization Methods|Local Update Strategies|Local SGD|85|BVR-L-SGD|ICML|2021|Bias-Variance Reduced Local SGD for Less Heterogeneous Federated Learning|[Link](https://arxiv.org/abs/2102.03198)|
|Distributed Optimization Methods|Decentralized Communication|Neighbor Communication Topology|86|A(DP)^2SGD|TPAMI|2021|A(DP)^2SGD: Asynchronous Decentralized Parallel Stochastic Gradient Descent with Differential Privacy|[Link](https://arxiv.org/abs/2008.09246)|
|First-Order Methods； Distributed Optimization Methods|Momentum-Enhanced SGD； Local Update Strategies； Distributed Hybrid Optimization|Accelerated Momentum； Local SGD； Local Momentum Updates； Compression & Local Updates|87|SQuARM-SGD|JSAIT|2021|SQuARM-SGD: Communication-Efficient Momentum SGD for Decentralized Optimization|[Link](https://arxiv.org/abs/2005.07041)|
|Distributed Optimization Methods|Decentralized Communication|Neighbor Communication Topology|88|LD-SGD|arXiv|2019|Communication-Efficient Local Decentralized SGD Methods|[Link](https://arxiv.org/abs/1910.09126)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Low-Rank Gradient Compression|89|SketchedAMSGrad|ICDM|2022|Communication-Efficient Adam-Type Algorithms for Distributed Data Mining|[Link](https://arxiv.org/abs/2210.07454)|
|Distributed Optimization Methods|Gradient Compression & Quantizatio|Sparsification Compression|90|SPARQ-SGD|TAC|2022|SPARQ-SGD: Event-Triggered and Compressed Communication in Decentralized Stochastic Optimization|[Link](https://arxiv.org/abs/1910.14280)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Adaptive Compression Level|91|AdaCGD|TMLR|2023|Adaptive Compression for Communication-Efficient Distributed Training|[Link](https://arxiv.org/abs/2211.00188)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Quantization Compression|92|0/1 Adam|ICLR|2023|Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam|[Link](https://arxiv.org/abs/2202.06009)|
|Zeroth-Order Methods|Distributed Zero-Order Optimization|Distributed Perturbation Sampling|93|ZoPro|CDC|2024|A Zeroth-Order Proximal Algorithm for Consensus Optimization|[Link](https://arxiv.org/abs/2406.09816)|
|Distributed Optimization Methods|Decentralized Communication|Neighbor Communication Topology|94|DAT-SGD|ICML|2025|Enhancing Parallelism in Decentralized Stochastic Convex Optimization|[Link](https://arxiv.org/abs/2506.00961)|
|Distributed Optimization Methods|Decentralized Communication|Distributed Consensus Optimization|95|LT-ADMM|arXiv|2025|Communication-Efficient Stochastic Distributed Learning|[Link](https://arxiv.org/abs/2501.13516)|
|Distributed Optimization Methods|Local Update Strategies|Local-Global Hybrid Updates|96|HybridSGD|arXiv|2025|Communication-Efficient, 2D Parallel Stochastic Gradient Descent for Distributed-Memory Optimization|[Link](https://arxiv.org/abs/2501.07526)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Compression Error Compensation|97|ADEF|arXiv|2025|Accelerated Distributed Optimization with Compression and Error Feedback|[Link](https://arxiv.org/abs/2503.08427)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Quantization Compression|98|TAH-QUANT|arXiv|2025|TAH-QUANT: Effective Activation Quantization in Pipeline Parallelism over Slow Network|[Link](https://arxiv.org/abs/2506.01352)|
|Distributed Optimization Methods|Decentralized Communication|Distributed Consensus Optimization|99|DEC-LOC|arXiv|2025|DES-LOC: Desynced Low Communication Adaptive Optimizers for Training Foundation Models|[Link](https://arxiv.org/abs/2505.22549)|
|Distributed Optimization Methods； Memory-Efficient Optimization Methods|Gradient Compression & Quantization； Low-Rank Methods|Quantization Compression； Low-Rank & Quantization|100|LQ-SGD|arXiv|2025|Trustworthy Efficient Communication for Distributed Learning using LQ-SGD Algorithm|[Link](https://arxiv.org/abs/2506.17974)|
|Distributed Optimization Methods|Gradient Compression & Quantization； Local Update Strategies|Adaptive Compression Level； Local-Global Hybrid Updates|101|Deco-SGD|arXiv|2025|DeCo-SGD: Joint Optimization of Delay Staleness and Gradient Compression Ratio for Distributed SGD|[Link](https://arxiv.org/abs/2507.17346)|
|Distributed Optimization Methods|Decentralized Communication|Distributed Consensus Optimization|102|DLAS-R-FTC|arXiv|2025|Distributed Optimization and Learning for Automated Stepsize Selection with Finite Time Coordination|[Link](https://arxiv.org/abs/2508.05887)|
|Distributed Optimization Methods|Local Update Strategies|Local SGD|103|SCAFFOLD|ICML|2020|SCAFFOLD: Stochastic Controlled Averaging for Federated Learning|[Link](https://arxiv.org/abs/1910.06378)|
|Distributed Optimization Methods|Local Update Strategies|Adaptive Local Steps|104|DP-PASGD|arXiv|2020|Differentially Private Federated Learning for Resource-Constrained Internet of Things|[Link](https://arxiv.org/abs/2003.12705)|
|Privacy-Preserving Optimization Methods|Gradient Noise Injection|Noise-Robust Optimization|105|DPZV|arXiv|2025|DPZV: Elevating the Tradeoff between Privacy and Utility in Zeroth-Order Vertical Federated Learning|[Link](https://arxiv.org/abs/2502.20565)|
|Privacy-Preserving Optimization Methods|Gradient Noise Injection； Federated Privacy Enhancement|Noise-Robust Optimization； Federated Noise Aggregation|106|DP-FedSAM|CVPR|2023|Make Landscape Flatter in Differentially Private Federated Learning|[Link](https://arxiv.org/abs/2303.11242)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|107|AdaFedAdam|IEEE T-MLCN|2024|ACCELERATING FAIR FEDERATED LEARNING: ADAPTIVE FEDERATED ADAM|[Link](https://arxiv.org/abs/2301.09357)|
|First-Order Methods|Hybrid Methods|Gradient Filtering Hybrid|108|FESS-GDA|AISTATS|2024|Stochastic Smoothed Gradient Descent Ascent for Federated Minimax Optimization|[Link](https://arxiv.org/abs/2311.00944)|
|Distributed Optimization Methods|Federated Learning Optimization|Personalized Federated Optimization|109|MM-PSGD|MMAsia|2024|Distributed Optimization over Block-Cyclic Data|[Link](https://arxiv.org/abs/2002.07454)|
|Distributed Optimization Methods|Federated Learning Optimization|Personalized Federated Optimization|109|MC-PSGD|MMAsia|2024|Distributed Optimization over Block-Cyclic Data|[Link](https://arxiv.org/abs/2002.07454)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|110|FedLion|ICASSP|2024|FEDLION: FASTER ADAPTIVE FEDERATED OPTIMIZATION WITH FEWER COMMUNICATION|[Link](https://arxiv.org/abs/2402.09941)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|111|FADAS|ICML|2024|FADAS: Towards Federated Adaptive Asynchronous Optimization|[Link](https://arxiv.org/abs/2407.18365)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|112|FLeNS|BigData|2024|FLeNS: Federated Learning with Enhanced Nesterov-Newton Sketch|[Link](https://arxiv.org/abs/2409.15216)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|113|FedRepOpt|ACCV|2024|FedRepOpt: Gradient Re-parametrized Optimizers in Federated Learning|[Link](https://arxiv.org/abs/2409.15898)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Second-Order Optimization|114|FAGH|arXiv|2024|FAGH: Accelerating Federated Learning with Approximated Global Hessian|[Link](https://arxiv.org/abs/2403.11041)|
|Distributed Optimization Methods|Decentralized Communication|Privacy-Preserving Decentralization|115|FedLAP-DP|arXiv|2023|FedLAP-DP: Federated Learning by Sharing Differentially Private Loss Approximations|[Link](https://arxiv.org/abs/2302.01068)|
|Distributed Optimization Methods|Federated Learning Optimization|Personalized Federated Optimization|116|FedIvon|arXiv|2024|Federated Learning with Uncertainty and Personalization via Efficient Second-order Optimization|[Link](https://arxiv.org/abs/2411.18385)|
|Distributed Optimization Methods|Federated Learning Optimization|Client Sampling Optimization|117|FedSTaS|arXiv|2024|FedSTaS: Client Stratification and Client Level Sampling for Efficient Federated Learning|[Link](https://arxiv.org/abs/2412.14226)|
|Distributed Optimization Methods|Local Update Strategies|Adaptive Local Steps|118|FedCET|arXiv|2025|Communication Efficient Federated Learning with Linear Convergence on Heterogeneous Data|[Link](https://arxiv.org/abs/2503.15804)|
|Distributed Optimization Methods|Federated Learning Optimization|Client Sampling Optimization|119|FAdamGC|arXiv|2025|Gradient Correction in Federated Learning with Adaptive Optimization|[Link](https://arxiv.org/abs/2502.02727)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Second-Order Optimization|120|FedCurv|arXiv|2025|Blockchain-Enabled Privacy-Preserving Second-Order Federated Edge Learning in Personalized Healthcare|[Link](https://arxiv.org/abs/2506.00416)|
||||121||arXiv|2025|Benchmarking optimizers for large language model pretraining|[Link](https://doi.org/10.48550/arXiv.2509.01440)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|122|Kuramoto-FedAvg|arXiv|2025|Kuramoto-FedAvg: Using Synchronization Dynamics to Improve Federated Learning Optimization under Statistical Heterogeneity|[Link](https://arxiv.org/abs/2505.19605)|
|Distributed Optimization Methods|Federated Learning Optimization|Federated Second-Order Optimization|123|pFedSOP|arXiv|2025|pFedSOP : Accelerating Training Of Personalized Federated Learning Using Second-Order Optimization|[Link](https://arxiv.org/abs/2506.07159)|
|Distributed Optimization Methods|Federated Learning Optimization|Client Sampling Optimization|124|FedOne|arXiv|2025|FedOne: Query-Efficient Federated Learning for Black-box Discrete Prompt Learning|[Link](https://arxiv.org/abs/2506.14929)|
|First-Order Methods|Learning Rate Scheduling|Scheduler-Free Adaptation|125|Step-Tuned SGD|Neural Process. Lett.|2022|Second-order step-size tuning of SGD for non-convex optimization|[Link](https://arxiv.org/abs/2103.03570)|
|First-Order Methods|Adaptive Learning Rate Methods|Stateless Adaptation|126|AEGDM|Ann. Appl. Math.|2022|AN ADAPTIVE GRADIENT METHOD WITH ENERGY AND MOMENTUM|[Link](https://arxiv.org/abs/2203.12191)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|127|AdaInject|ITAI|2022|AdaInject: Injection Based Adaptive Gradient Descent Optimizers for Convolutional Neural Networks|[Link](https://arxiv.org/abs/2109.12504)|
|First-Order Methods|Loss Landscape Optimization|Weight Averaging|128|look around|NeurIPS|2023|Lookaround Optimizer: k steps around, 1 step average|[Link](https://arxiv.org/abs/2306.07684)|
|First-Order Methods|Momentum-Enhanced SGD|Dynamic Momentum Weight|129|SGDF|arXiv|2023|Signal Processing Meets SGD: From Momentum to Filter|[Link](https://arxiv.org/abs/2311.02818)|
|First-Order Methods|Gradient Normalization & Clipping|Noise-Robust Normalization|130|NIGT|ICML|2020|Momentum Improves Normalized SGD|[Link](https://proceedings.mlr.press/v119/cutkosky20b.html)|
|First-Order Methods|Loss Landscape Optimization|Sharpness-Aware Minimization (SAM)|131|SAM|ICLR|2021|Sharpness-Aware Minimization for Efficiently Improving Generalization|[Link](https://openreview.net/forum?id=6Tm1mposlrM)|
|First-Order Methods|Loss Landscape Optimization|Sharpness-Aware Minimization (SAM)|132|ESAM|ICLR|2022|EFFICIENT SHARPNESS-AWARE MINIMIZATION FOR IMPROVED TRAINING OF NEURAL NETWORKS|[Link](https://arxiv.org/abs/2110.03141)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|133|AdaSAM|Neural Networks|2024|AdaSAM: Boosting Sharpness-Aware Minimization with Adaptive Learning Rate and Momentum for Training Deep Neural Networks|[Link](https://arxiv.org/abs/2303.00565)|
|First-Order Methods|Loss Landscape Optimization|Curvature-Guided Landscape Exploration|134|GAM|CVPR|2023|Gradient Norm Aware Minimization Seeks First-Order Flatness and ImprovesGeneralization|[Link](https://arxiv.org/abs/2303.03108)|
|First-Order Methods|Loss Landscape Optimization|Sharpness-Aware Minimization (SAM)|135|AE-SAM|ICLR|2023|AN ADAPTIVE POLICY TO EMPLOY SHARPNESS-AWARE MINIMIZATION|[Link](https://arxiv.org/abs/2304.14647)|
|First-Order Methods|Loss Landscape Optimization|Sharpness-Aware Minimization (SAM)|136|SAMPa|NeurIPS|2024|SAMPa: Sharpness-aware Minimization Parallelized|[Link](https://arxiv.org/abs/2410.10683)|
|First-Order Methods|Loss Landscape Optimization|Momentum Landscape Adaptation|137|MSAM|arXiv|2024|Momentum-SAM: Sharpness Aware Minimization without Computational Overhead|[Link](https://arxiv.org/abs/2401.12033)|
|First-Order Methods|Loss Landscape Optimization|Multi-Step Ascent SAM|138||ICML|2024|Lookbehind-SAM: k steps back, 1 step forward|[Link](https://arxiv.org/abs/2307.16704)|
|First-Order Methods|Loss Landscape Optimization|Noise Injection Enhancement|139|F-SAM|CVPR|2024|Friendly Sharpness-Aware Minimization|[Link](https://arxiv.org/abs/2403.12350)|
|First-Order Methods|Loss Landscape Optimization|Noise Injection Enhancement|140|FGSAM|NeurIPS|2024|Fast Graph Sharpness-Aware Minimization for Enhancing and Accelerating Few-Shot Node Classification|[Link](https://arxiv.org/abs/2410.16845)|
|First-Order Methods|Loss Landscape Optimization|Renormalized Gradient Norm SAM|141|SSAM|JMLR|2025|Stabilizing Sharpness-aware Minimization Through A Simple Renormalization Strategy|[Link](https://arxiv.org/abs/2401.07250)|
|First-Order Methods|Gradient Normalization & Clipping|Mean-Removal Normalization|142|GCSAM|arXiv|2025|GCSAM: Gradient Centralized Sharpness Aware Minimization|[Link](https://arxiv.org/abs/2501.11584)|
|First-Order Methods|Loss Landscape Optimization|Sharpness-Aware Minimization (SAM)|143|AsyncSAM|arXiv|2025|ASYNCHRONOUS SHARPNESS-AWARE MINIMIZATION FOR FAST AND ACCURATE DEEP LEARNING|[Link](https://arxiv.org/abs/2503.11147)|
|First-Order Methods|Loss Landscape Optimization|Sharpness-Aware Minimization (SAM)|144|LightSAM|arXiv|2025|LightSAM: Parameter-Agnostic Sharpness-Aware Minimization|[Link](https://arxiv.org/abs/2505.24399)|
|Zeroth-Order Methods|Memory-efficient Methods|Quantized Zeroth-Order Finetuning|145|ZOQO|ICASSP|2025|ZOQO: Zero-Order Quantized Optimization|[Link](https://arxiv.org/abs/2501.06736)|
|Zeroth-Order Methods|Zeroth-First Order Hybrid|Layer-Wise Hybrid|146|ELASTICZO|arXiv|2025|ELASTICZO: A MEMORY-EFFICIENT ON-DEVICE LEARNING WITH COMBINED ZEROTH- AND FIRST-ORDER OPTIMIZATION|[Link](https://arxiv.org/abs/2501.04287)|
|Zeroth-Order Methods|Adaptive Method|Momentum-based Adaptive|147|R-AdaZO|ICML|2025|Refining Adaptive Zeroth-Order Optimization at Ease|[Link](https://arxiv.org/abs/2502.01014)|
|First-Order Methods|Adaptive Learning Rate Methods||148|AdamW / SGDW|ICLR|2019|Decoupled Weight Decay Regularization|[Link](https://arxiv.org/abs/1711.05101)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation； Prediction Deviation Adaptation|149|AdaBelief|NeurIPS|2020|AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients|[Link](https://arxiv.org/abs/2010.07468)|
|First-Order Methods|Adaptive Learning Rate Methods|Decoupled Learning Rate and Adaptability|150|AvaGrad|CVPR|2021|Domain-independent Dominance of Adaptive Methods|[Link](https://arxiv.org/abs/1912.01823)|
|First-Order Methods|Adaptive Learning Rate Methods|Prediction Deviation Adaptation|151|Aida|TMLR|2023|A DNN Optimizer that Improves over AdaBelief by Suppression of the Adaptive Stepsize Range|[Link](https://arxiv.org/abs/2203.13273)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|152|AdaDelta|arXiv|2012|ADADELTA:ANADAPTIVELEARNINGRATEMETHOD|[Link](https://arxiv.org/abs/1212.5701)|
|First-Order Methods|Momentum-Enhanced SGD|Double-momentum mechanism|153|YOGI|NeurIPS|2018|Adaptive Methods for Nonconvex Optimization|[Link](http://papers.neurips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf)|
|Memory-Efficient Optimization Methods|Low-Memory Optimizer Design； Optimizer State Compression|Compression&Approximation of States； State Sharing|154|Adafactor|ICML|2018|Adafactor: Adaptive Learning Rates with Sublinear Memory Cost|[Link](https://arxiv.org/abs/1804.04235)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|155|NAdam|ICLR Workshop|2016|Incorporating Nesterov Momentum into Adam|[Link](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)|
|First-Order Methods|Adaptive Learning Rate Methods||156|LaProp|arXiv|2020|LaProp: Separating Momentum and Adaptivity in Adam|[Link](https://arxiv.org/abs/2002.04839)|
|First-Order Methods|Adaptive Learning Rate Methods|Layer-Wise Adaptation|157|NovoGrad|arXiv|2019|Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks|[Link](https://arxiv.org/abs/1905.11286)|
|First-Order Methods|Hybrid Methods|Gradient Filtering Hybrid|158|VR-SGD|TKDE|2018|VR-SGD: A Simple Stochastic Variance Reduction Method for Machine Learning|[Link](https://arxiv.org/abs/1802.09932)|
|First-Order Methods|Adaptive Learning Rate Methods||159|QHadam|ICLR|2019|QUASI-HYPERBOLIC MOMENTUM AND ADAM FORDEEP FOR DEEP LEARNING|[Link](https://arxiv.org/abs/1810.06801)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|160|RAdam|ICLR|2020|On the Variance of the Adaptive Learning Rate and Beyond|[Link](https://arxiv.org/abs/1908.03265)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|161|Adam+|arXiv|2020|Adam+: A Stochastic Method with Adaptive Variance Reduction|[Link](https://arxiv.org/abs/2011.11985)|
|Zeroth-Order Methods|Zeroth-First Order Hybrid； Variance Reduction|Variance Reduction Hybrid； Snapshot Variance Reduction|162|VAMO|arXiv|2025|VAMO: Efficient Large-Scale Nonconvex Optimization via Adaptive Zeroth Order Variance Reduction|[Link](https://arxiv.org/abs/2505.13954)|
|Memory-Efficient Optimization Methods|Low-Rank Methods|Projection\&Adjustment|163|Adapprox|arXiv|2024|Adapprox: Adaptive Approximation in Adam Optimization via Randomized Low-Rank Matrices|[Link](https://arxiv.org/abs/2403.14958)|
|Distributed Optimization Methods|Gradient Compression & Quantization|Quantization Compression|164|signSGD|ICML|2018|signSGD: Compressed Optimisation for Non-Convex Problems|[Link](https://proceedings.mlr.press/v80/bernstein18a.html)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Automated Discovery&Theoretical Derivation|165||ICML|2017|Neural Optimizer Search with Reinforcement Learning|[Link](https://arxiv.org/abs/1709.07417)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Automated Discovery&Theoretical Derivation|166||ICLR|2017|Learning to optimize.|[Link](https://arxiv.org/abs/1606.01885)|
|First-Order Methods； Other Specialized Optimization Methods|Adaptive Learning Rate Methods； Auto-Designed Optimizers|Momentum-based Adaptive; Automated Discovery&Theoretical Derivation|167|Lion|NeurIPS|2023|Symbolic Discovery of Optimization Algorithms|[Link](https://arxiv.org/abs/2302.06675)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|168|Adan|TPAMI|2024|Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models|[Link](https://arxiv.org/abs/2208.06677)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|169|AdaGrad|JMLR|2011|Adaptive Subgradient Methods for Online Learning and Stochastic Optimization|[Link](https://jmlr.org/papers/v12/duchi11a.html)|
|First-Order Methods|Preconditioned Gradient Methods|Two Metrics' Preconditioner|170|Shampoo|ICML|2018|Shampoo: Preconditioned Stochastic Tensor Optimization|[Link](https://arxiv.org/abs/1802.09568)|
|First-Order Methods|Momentum-Enhanced SGD； Hybrid Methods|Accelerated Momentum； Multi-Objective Hybrid|171|MADGRAD|JMLR|2022|Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization|[Link](https://arxiv.org/abs/2101.11075)|
|First-Order Methods； Memory-Efficient Optimization Methods|Preconditioned Gradient Methods； Low-Memory Optimizer Design|Two Metrics' Preconditioner； Compression&Approximation of States|172|4-bit shampoo|NeurIPS|2024|4-bit Shampoo for Memory-Efficient Network Training|[Link](https://arxiv.org/abs/2405.18144)|
|First-Order Methods|Adaptive Learning Rate Methods|Layer-Wise Adaptation|173|Muon|Blog|2024|Muon: An optimizer for hidden layers in neural networks|[Link](https://kellerjordan.github.io/posts/muon/)|
|First-Order Methods|Hybrid Methods|Projection Gradient Hybrid|174|ADAGB2|arXiv|2025|Fast Stochastic Second-Order Adagrad for Nonconvex Bound-Constrained Optimization|[Link](https://arxiv.org/abs/2505.06374)|
|First-Order Methods|Preconditioned Gradient Methods|Two Metrics' Preconditioner|175|Splus|arXiv|2025|A Stable Whitening Optimizer for Efficient Neural Network Training|[Link](https://arxiv.org/abs/2506.07254)|
|First-Order Methods|Preconditioned Gradient Methods|Single Metric's Preconditioner|176|ASGO|arXiv|2025|ASGO: Adaptive Structured Gradient Optimization|[Link](https://arxiv.org/abs/2503.20762)|
|Second-Order Methods|Hessian Approximation & Estimation|Diagonal Hessian Approximation|177|CRNAS|arXiv|2024|Novel Optimization Techniques for Parameter Estimation|[Link](https://arxiv.org/abs/2407.04235)|
|First-Order Methods|Adaptive Learning Rate Methods|Hybrid Adaptive Strategy|178|AdaMuon|arXiv|2025|ADAMUON: ADAPTIVE MUON OPTIMIZER|[Link](https://arxiv.org/abs/2507.11005)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|179|Accelerated GRAAL|arXiv|2025|NESTEROV FINDS GRAAL: OPTIMAL AND ADAPTIVE GRADIENT METHOD FOR CONVEX OPTIMIZATION|[Link](https://arxiv.org/abs/2507.09823)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|180|eagle|arXiv|2025|EAGLE: EARLY APPROXIMATED-GRADIENT-BASED LEARNING RATE ESTIMATOR|[Link](https://arxiv.org/abs/2502.01036)|
|First-Order Methods|Momentum-Enhanced SGD； Adaptive Learning Rate Methods|Double-momentum mechanism；Momentum-based Adaptive|181|MARS|ICML|2025|MARS: Unleashing the Power of Variance Reduction for Training Large Models|[Link](https://arxiv.org/abs/2411.10438)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Evolutionary Strategies&Meta-Adaptive Learning|182|GADAM|arXiv|2018|GADAM: Genetic-Evolutionary ADAM for Deep Neural Network Optimization|[Link](https://arxiv.org/abs/1805.07500)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|183|HAdam|NeurIPS|2019|On Higher-order Moments in Adam|[Link](https://arxiv.org/abs/1910.06878)|
|First-Order Methods|Adaptive Learning Rate Methods|Hybrid Adaptive Strategy|184|diffGrad|IEEE TNNLS|2019|diffGrad: An Optimization Method for Convolutional Neural Networks|[Link](https://arxiv.org/abs/1909.11015)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|185|AdamBS|NeurIPS|2020|Adam with Bandit Sampling for Deep Learning|[Link](https://arxiv.org/abs/2010.12986)|
|First-Order Methods|Momentum-Enhanced SGD|Dynamic Momentum Weight|186|DEAM|ASONAM|2020|DEAM: Adaptive Momentum with Discriminative Weight for Stochastic Optimization|[Link](https://arxiv.org/abs/1907.11307)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Evolutionary Strategies&Meta-Adaptive Learning|187|BGADAM|IJCNN|2021|BGADAM: Boosting based Genetic-Evolutionary ADAM for Neural Network Optimization|[Link](https://arxiv.org/abs/1908.08015)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|188|Madam|ECML|2021|MaxVA: Fast Adaptation of Step Sizes by Maximizing Observed Variance of Gradients|[Link](https://arxiv.org/abs/2006.11918)|
|First-Order Methods|Learning Rate Scheduling|Gradient Angle Scheduling|189|ACMo|AAAI|2021|ACMO: ANGLE-CALIBRATED MOMENT METHODS FOR STOCHASTIC OPTIMIZATION|[Link](https://arxiv.org/abs/2006.07065)|
|First-Order Methods|Gradient Normalization & Clipping|Mean-Removal Normalization|190|AdamMC|CVMI|2023|Moment Centralization based Gradient Descent Optimizers for Convolutional Neural Networks|[Link](https://arxiv.org/abs/2207.09066)|
|First-Order Methods； Memory-Efficient Optimization Methods|Learning Rate Scheduling； Stateless Optimization Methods|Gradient Angle Scheduling； Parameter Characteristic-Driven Updates|191|AdaBFE|arXiv|2022|BFE and AdaBFE: A New Approach in Learning Rate Automation for Stochastic Optimization|[Link](https://arxiv.org/abs/2207.02763)|
|First-Order Methods|Learning Rate Scheduling|Scheduler-Free Adaptation|192|Amos|arXiv|2022|Amos: AN ADAM-STYLE OPTIMIZER WITH ADAPTIVE WEIGHT DECAY TOWARDS MODEL-ORIENTED SCALE|[Link](https://arxiv.org/abs/2210.11693)|
|First-Order Methods|Gradient Normalization & Clipping|Layer-Wise Gradient Normalization|193|MultiAdam|ICML|2023|MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training of Physics-informed Neural Networks|[Link](https://arxiv.org/abs/2306.02816)|
|First-Order Methods； Other Specialized Optimization Methods|Gradient Normalization & Clipping； Robust Optimization|Element-Wise Gradient Scaling； Noise-Robust Normalization； Noise-Robust Gradients|194|AdaNorm|WACV|2023|AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs|[Link](https://arxiv.org/abs/2210.06364)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Evolutionary Strategies&Meta-Adaptive Learning|195|WarpAdam|arXiv|2024|WarpAdam: A new Adam optimizer based on Meta-Learning approach|[Link](https://arxiv.org/abs/2409.04244)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|196|ADOPT|NeurIPS|2024|ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate|[Link](https://arxiv.org/abs/2411.02853)|
|First-Order Methods|Adaptive Learning Rate Methods； Hybrid Methods|Second-Order Moment Adaptation； Multi-Objective Hybrid|197|SET-adam|ECML|2024|On Suppressing Range of Adaptive Stepsizes of Adam to Improve Generalisation Performance|[Link](https://arxiv.org/abs/2302.01029)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|198|VSGD|TMLR|2025|Variational Stochastic Gradient Descent for Deep Neural Networks|[Link](https://arxiv.org/abs/2404.06549)|
|First-Order Methods|Learning Rate Scheduling|Scheduler-Free Adaptation|199|Adam++|arXiv|2024|Towards Simple and Provable Parameter-Free Adaptive Gradient Methods|[Link](https://arxiv.org/abs/2412.19444)|
|First-Order Methods|Loss Landscape Optimization|Curvature-Guided Landscape Exploration|200|MIAdam|AAAI|2025|A Method for Enhancing Generalization of Adam by Multiple Integrations|[Link](https://arxiv.org/abs/2412.12473)|
|First-Order Methods|Momentum-Enhanced SGD|Scheduled Momentum Reset|201|Adam-Real|NeurIPS|2024|Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps|[Link](https://arxiv.org/abs/2412.17113)|
|First-Order Methods|Hybrid Methods|Projection Gradient Hybrid|202|PAdamP|CAMMIC|2025|ADAPTIVE MOMENT ESTIMATION OPTIMIZATION ALGORITHM USING PROJECTION GRADIENT FOR DEEP LEARNING|[Link](https://arxiv.org/abs/2503.10005)|
|First-Order Methods|Momentum-Enhanced SGD|Momentum Damping Mechanism|203|VRAdam|arXiv|2025|A Physics-Inspired Optimizer: Velocity Regularized Adam|[Link](https://arxiv.org/abs/2505.13196)|
|First-Order Methods； Memory-Efficient Optimization Methods|Adaptive Learning Rate Methods； Low-Memory Optimizer Design； Stateless Optimization Methods|Stateless Adaptation； Structural Redesign； Parameter Characteristic-Driven Updates|204|AlphaGrad|arXiv|2025|AlphaGrad: Non-Linear Gradient Normalization Optimizer|[Link](https://arxiv.org/abs/2504.16020)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Automated Discovery&Theoretical Derivation|205|KO|arXiv|2025|KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches|[Link](https://arxiv.org/abs/2505.14777)|
|First-Order Methods|Adaptive Learning Rate Methods|Hybrid Adaptive Strategy|206|EXADAM|arXiv|2025|EXADAM: THE POWER OF ADAPTIVE CROSS-MOMENTS|[Link](https://arxiv.org/abs/2412.20302)|
|First-Order Methods|Learning Rate Scheduling|Gradient Angle Scheduling|207|HGM|arXiv|2025|Hindsight-Guided Momentum (HGM) Optimizer: An Approach to Adaptive Learning Rates|[Link](https://arxiv.org/abs/2506.22479)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|208|CAdam|arXiv|2025|CAdam: Confidence-Based Optimization for Online Learning|[Link](https://arxiv.org/abs/2411.19647)|
|First-Order Methods|Loss Landscape Optimization|Curvature-Guided Landscape Exploration|209|DEO|arXiv|2025|Dimer-Enhanced Optimization: A First-Order Approach to Escaping Saddle Points in Neural Network Training|[Link](https://arxiv.org/abs/2507.19968)|
|First-Order Methods|Learning Rate Scheduling|Stability-Aware Adaptive Scheduling|210|LyAm|arXiv|2025|LyAm: Robust Non-Convex Optimization for Stable Learning in Noisy Environments|[Link](https://arxiv.org/abs/2507.11262)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|211|ZetA|arXiv|2025|ZETA: A HYBRID OPTIMIZER COMBINING RIEMANN ZETA SCALING WITH ADAM FOR ROBUST DEEP LEARNING|[Link](https://arxiv.org/abs/2508.02719)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|212|MSVAG|ICML|2018|DissectingAdam:TheSign,MagnitudeandVarianceofStochasticGradients|[Link](https://arxiv.org/abs/1705.07774)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|213|NosAdam|IJCAI|2019|Nostalgic Adam: Weighting more of the past gradients when designing the adaptive learning rate|[Link](https://arxiv.org/abs/1805.07557)|
|First-Order Methods|Adaptive Learning Rate Methods|Dynamic Epsilon Adjustment|214|EAdam|arXiv|2020|EAdam Optimizer: How ∈ Impact Adam|[Link](https://arxiv.org/abs/2011.02150)|
|First-Order Methods|Adaptive Learning Rate Methods|Hybrid Adaptive Strategy|215|AdaL|CoRR|2021|AdaL: Adaptive Gradient Transformation Contributes to Convergences and Generalizations|[Link](https://arxiv.org/abs/2107.01525)|
|First-Order Methods|Momentum-Enhanced SGD|Momentum-Gradient Alignment|216|AngularGrad|arXiv|2021|AngularGrad: A New Optimization Technique for Angular Convergence of Neural Networks|[Link](https://arxiv.org/abs/2105.10190)|
|First-Order Methods|Adaptive Learning Rate Methods|Bias Correction Rules Adaptaion|217|AdamD|arXiv|2021|AdamD: Improved bias-correction in Adam|[Link](https://arxiv.org/abs/2110.10828)|
|First-Order Methods|Adaptive Learning Rate Methods|Hybrid Adaptive Strategy|218|AdamFamily|arXiv|2022|AdaFamily: A family of Adam-like adaptive gradient methods|[Link](https://arxiv.org/abs/2203.01603)|
|First-Order Methods|Adaptive Learning Rate Methods； Hybrid Methods|Dynamic Epsilon Adjustment； Multi-Objective Hybrid|219|FAdam|arXiv|2024|FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information|[Link](https://arxiv.org/abs/2405.12807)|
|Second-Order Methods|Fisher Information Matrix Application|Diagonal Fisher Approximation|220|RACS|arXiv|2025|Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension|[Link](https://arxiv.org/abs/2502.07752)|
|First-Order Methods|Adaptive Learning Rate Methods|Layer-Wise Adaptation|221|CaAdam|arXiv|2024|CaAdam: Improving Adam optimizer using connection aware methods|[Link](https://arxiv.org/abs/2410.24216)|
|Other Specialized Optimization Methods|Auto-Designed Optimizers|Evolutionary Strategies&Meta-Adaptive Learning|222|HyperAdam|AAAI|2019|HyperAdam: A Learnable Task-Adaptive Adam for Network Training|[Link](https://arxiv.org/abs/1811.08996)|
|First-Order Methods|Adaptive Learning Rate Methods|Layer-Wise Adaptation|223|LARS|arXiv|2017|Large batch training of Convolutional Network|[Link](https://arxiv.org/abs/1708.03888)|
|First-Order Methods|Momentum-Enhanced SGD|Momentum Damping Mechanism|224|PIDOptimizer|CVPR|2018|A PID Controller Approach for Stochastic Optimization of Deep Networks|[Link](https://openaccess.thecvf.com/content_cvpr_2018/html/An_A_PID_Controller_CVPR_2018_paper.html)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|225|Lookahead|NeurIPS|2019|Lookahead Optimizer: k steps forward, 1 step back|[Link](https://arxiv.org/abs/1907.08610)|
|First-Order Methods|Learning Rate Scheduling|Element-Wise Learning Rate Scheduling|226|AdaBound|ICLR|2019|Adaptive Gradient Methods with Dynamic Bound of Learning Rate|[Link](https://arxiv.org/abs/1902.09843)|
|First-Order Methods|Adaptive Learning Rate Methods； Learning Rate Scheduling|Layer-Wise Adaptation； Batch-Aware Scheduling|227|LAMB|ICLR|2020|Large Batch Optimization for Deep Learning: Training BERT in 76 minutes|[Link](https://arxiv.org/abs/1904.00962)|
|First-Order Methods|Momentum-Enhanced SGD|Momentum Damping Mechanism|228|SNGM|Sci. China Inf. Sci.|2024|Stochastic Normalized Gradient Descent with Momentum for Large-Batch Training|[Link](https://arxiv.org/abs/2007.13985)|
|First-Order Methods|Momentum-Enhanced SGD； Hybrid Methods|Momentum Damping Mechanism； Projection Gradient Hybrid|229|AdamP / SGDP|ICLR|2021|AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights|[Link](https://arxiv.org/abs/2006.08217)|
|First-Order Methods|Adaptive Learning Rate Methods|Hybrid Adaptive Strategy|230|ACProp|NeurIPS|2021|Momentum Centering and Asynchronous Update for Adaptive Gradient Methods|[Link](https://arxiv.org/abs/2110.05454)|
|Memory-Efficient Optimization Methods|Low-Memory Optimizer Design|Structural Redesign|231|tpSGD|SMC|2023|Learning with Local Gradients at the Edge|[Link](https://arxiv.org/abs/2208.08503)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|232|AGD|NeurIPS|2023|AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix|[Link](https://arxiv.org/abs/2312.01658)|
|First-Order Methods|Learning Rate Scheduling|Scheduler-Free Adaptation|233|Schedule-Free|NeurIPS|2024|The Road Less Scheduled|[Link](https://arxiv.org/abs/2405.15682)|
|First-Order Methods|Adaptive Learning Rate Methods； Learning Rate Scheduling|Stateless Adaptation； Scheduler-Free Adaptation|234|AUTODROP|UAI|2024|AUTODROP: TRAINING DEEP LEARNING MODELS WITH AUTOMATIC LEARNING RATE DROP|[Link](https://arxiv.org/abs/2111.15317)|
|First-Order Methods|Momentum-Enhanced SGD|Momentum-Gradient Alignment|235|Cautious Optimizers|arXiv|2024|Cautious Optimizers: Improving Training with One Line of Code|[Link](https://arxiv.org/abs/2411.16085)|
|First-Order Methods； Other Specialized Optimization Methods|Hybrid Methods； Auto-Designed Optimizers|Gradient Smoothing Hybrid； Automated Discovery&Theoretical Derivation|236|AGS-GD|arXiv|2024|Anisotropic Gaussian Smoothing for Gradient-based Optimization|[Link](https://arxiv.org/abs/2411.11747)|
|First-Order Methods； Memory-Efficient Optimization Methods|Adaptive Learning Rate Methods； Optimizer State Compression|Neuron-Level Adaptation； State Sharing|237|ADAACT|ICDMW|2024|AN ADAPTIVE METHOD STABILIZING ACTIVATIONS FOR ENHANCED GENERALIZATION|[Link](https://arxiv.org/abs/2506.08353)|
|First-Order Methods|Learning Rate Scheduling|Loss-Sensitive Scheduling|238|DecGD|Machine Learning|2025|A New Adaptive Gradient Method with Gradient Decomposition|[Link](https://arxiv.org/abs/2107.08377)|
|First-Order Methods|Momentum-Enhanced SGD|Scheduled Momentum Reset|239|LazyOptimizer|Blog|2019||[Link](https://github.com/bojone/keras_lazyoptimizer)|
|First-Order Methods|Loss Landscape Optimization|Curvature-Guided Landscape Exploration|240|SKA-SGD|arXiv|2025|STREAMING KRYLOV-ACCELERATED STOCHASTIC GRADIENT DESCENT|[Link](https://arxiv.org/abs/2505.07046)|
|Second-Order Methods|Quasi-Newton Methods|Stochastic BFGS|241|FUSE-PV|CAI|2025|FUSE: First-Order and Second-Order Unified SynthEsis in Stochastic Optimization|[Link](https://arxiv.org/abs/2503.04204)|
|First-Order Methods|Momentum-Enhanced SGD|Accelerated Momentum|242|SGDO|arXiv|2025|Overshoot: Taking advantage of future gradients in momentum-based stochastic optimization|[Link](https://arxiv.org/abs/2501.09556)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|243|MoMo|ICML|2024|MoMo: Momentum Models for Adaptive Learning Rates|[Link](https://arxiv.org/abs/2305.07583)|
|Other Specialized Optimization Methods|Robust Optimization|Structure-Aware Optimization|244|BC-ADMM|arXiv|2025|BC-ADMM: An Efficient Non-convex Constrained Optimizer with Robotic Applications|[Link](https://arxiv.org/abs/2504.05465)|
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|245|Adam-Power|arXiv|2025|GradPower: Powering Gradients for Faster Language Model Pre-Training|[Link](https://arxiv.org/abs/2505.24275)|
|Distributed Optimization Methods|Local Update Strategies|Adaptive Local Steps|246|AbsSADMM|arXiv|2025|Stochastic ADMM with batch size adaptation for nonconvex nonsmooth optimization|[Link](https://arxiv.org/abs/2505.06921)|
|First-Order Methods|Preconditioned Gradient Methods|Single Metric's Preconditioner|247|Hessian-aware Scaling|arXiv|2025|First-ish Order Methods: Hessian-aware Scalings of Gradient Descent|[Link](https://arxiv.org/abs/2502.03701)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|248|NIRMAL|arXiv|2025|COMPARATIVE ANALYSIS OF NOVEL NIRMAL OPTIMIZER AGAINST ADAM AND SGD WITH MOMENTUM|[Link](https://arxiv.org/abs/2508.04293)|
|Zeroth-Order Methods|Variance Reduction|Snapshot Variance Reduction|249|VR-SZD|arXiv|2025|A Structured Proximal Stochastic Variance Reduced Zeroth-order Algorithm|[Link](https://arxiv.org/abs/2506.23758)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|250|Grams|ICLR Workshop|2025|Grams: Gradient Descent with Adaptive Momentum Scaling|[Link](https://arxiv.org/abs/2412.17107)|
|First-Order Methods|Momentum-Enhanced SGD|Frequency Domain Momentum Analysis|251|FSGDM|ICLR|2025|ON THE PERFORMANCE ANALYSIS OF MOMENTUM METHOD: A FREQUENCY DOMAIN PERSPECTIVE|[Link](https://arxiv.org/abs/2411.19671)|
|First-Order Methods|Momentum-Enhanced SGD|Accelerated Momentum|252|RSGDM|CCSB|2024|Reducing Bias in Deep Learning Optimization: The RSGDM Approach|[Link](https://arxiv.org/abs/2409.15314)|
|First-Order Methods|Momentum-Enhanced SGD|Double-momentum mechanism|253|AdEMAMix|ICLR|2025|THE ADEMAMIX OPTIMIZER:BETTER, FASTER, OLDER|[Link](https://arxiv.org/abs/2409.03137)|
|First-Order Methods|Adaptive Learning Rate Methods|Kalman filtering based|254|RLEKF|AAAI|2023|RLEKF: An Optimizer for Deep Potential with Ab Initio Accuracy|[Link](https://arxiv.org/abs/2212.06989)|
|Second-Order Methods|Fisher Information Matrix Application|Block-Diagonal Kronecker Approximation|255|K-FAC|ICML|2015|Optimizing Neural Networks with Kronecker-factored Approximate Curvature|[Link](https://arxiv.org/abs/1503.05671)|
|Second-Order Methods|Quasi-Newton Methods|Stochastic BFGS|256|SpiderSQN|IFAC-PapersOnLine|2020|A FAST QUASI-NEWTON-TYPE METHOD FOR LARGESCALE STOCHASTIC OPTIMISATION|[Link](https://arxiv.org/abs/2004.06479)|
|Second-Order Methods|Hessian Approximation & Estimation|Stochastic Hessian Sampling|257|SGN|arXiv|2020|On the Promise of the Stochastic Generalized Gauss-Newton Method for Training DNNs|[Link](https://arxiv.org/abs/2006.02409)|
|Second-Order Methods|Hessian Approximation & Estimation； Second-Order Moment Fusion|Diagonal Hessian Approximation； Noise-Robust Second-Order Momentum|258|AdaHessian|AAAI|2021|AdaHessian: An Adaptive Second Order Optimizer for Machine Learning|[Link](https://arxiv.org/abs/2006.00719)|
|Second-Order Methods|Fisher Information Matrix Application|Trace-Preserving Fisher Approximation|259|TKFAC|AAAI|2021|A Trace-restricted Kronecker-Factored Approximation to Natural Gradient|[Link](https://arxiv.org/abs/2011.10741)|
|First-Order Methods|Hybrid Methods|Multi-Objective Hybrid|260|GDA-AM|ICLR|2022|GDA-AM: On the effectiveness of solving minimax optimization via Anderson Acceleration|[Link](https://arxiv.org/abs/2110.02457)|
|Second-Order Methods|Hessian Approximation & Estimation|Gradient Difference Estimation|261|SGDHess|NeurIPS|2022|Better SGD using Second-order Momentum|[Link](https://arxiv.org/abs/2103.03265)|
|Second-Order Methods|Quasi-Newton Methods|Stochastic BFGS； Low-Memory Quasi-Newton|262|mL-BFGS|TMLR|2023|mL-BFGS: A Momentum-based L-BFGS for Distributed Large-Scale Neural Network Optimization|[Link](https://arxiv.org/abs/2307.13744)|
|Second-Order Methods|Hessian Approximation & Estimation|Stochastic Hessian Sampling|263|SkechySGD|SIAM|2024|SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates|[Link](https://arxiv.org/abs/2211.08597)|
|Second-Order Methods； Privacy-Preserving Optimization Methods； Memory-Efficient Optimization Methods|Hessian Approximation & Estimation； Curvature-Guided Preconditioning； Second-Order Moment Fusion； Privacy-Aware Gradient Clipping； Stateless Optimization Methods|Diagonal Hessian Approximation； Hessian Diagonal Preconditioning； Noise-Robust Second-Order Momentum； Real-Time Curvature Estimation|264|sophia|ICLR|2024|Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training|[Link](https://arxiv.org/abs/2305.14342)|
|Second-Order Methods； Distributed Optimization Methods|Hessian Approximation & Estimation； Federated Learning Optimization|Diagonal Hessian Approximation； Federated Second-Order Optimization|265|Fed-Sophia|ICC|2024|Fed-Sophia: A Communication-Efficient Second-Order Federated Learning Algorithm|[Link](https://arxiv.org/abs/2406.06655)|
|Second-Order Methods|Curvature-Guided Preconditioning|Hessian Diagonal Preconditioning|266|OptiQ|arXiv|2024|Second-Order Optimization via Quiescence|[Link](https://arxiv.org/abs/2410.08033)|
|Second-Order Methods|Fisher Information Matrix Application|Diagonal Fisher Approximation|267|SOAA|arXiv|2024|EFFICIENT SECOND-ORDER NEURAL NETWORK OPTIMIZATION VIA ADAPTIVE TRUST REGION METHODS|[Link](https://arxiv.org/abs/2410.02293)|
|Second-Order Methods|Quasi-Newton Methods|Low-Memory Quasi-Newton|268|K-BFGS and K-BFGS(L),|NeurIPS|2020|Practical Quasi-Newton Methods for Training Deep Neural Networks|[Link](https://arxiv.org/abs/2006.08877)|
|First-Order Methods|Preconditioned Gradient Methods|Single Metric's Preconditioner|269|NYSACT|BigData|2024|NYSACT: A SCALABLE PRECONDITIONED GRADIENT DESCENT USING NYSTRÖM APPROXIMATION|[Link](https://arxiv.org/abs/2506.08360)|
|Second-Order Methods|Hessian Approximation & Estimation|Diagonal Hessian Approximation|270|HesScale|ICML|2024|Revisiting Scalable Hessian Diagonal Approximations for Applications in Reinforcement Learning|[Link](https://arxiv.org/abs/2406.03276)|
|Second-Order Methods|Hessian Approximation & Estimation|Block Hessian Approximation|271|Athena|arXiv|2024|Athena: Efficient Block-Wise Post-Training Quantization for Large Language Models Using Second-Order Matrix Derivative Information|[Link](https://arxiv.org/abs/2405.17470)|
|Second-Order Methods|Hessian Approximation & Estimation|Diagonal Hessian Approximation|272|SASSHA|ICML|2025|SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation|[Link](https://arxiv.org/abs/2502.18153)|
|Second-Order Methods|Fisher Information Matrix Application|Diagonal Fisher Approximation； Block-Diagonal Kronecker Approximation|273|AdaFisher|ICLR|2025|ADAFISHER: ADAPTIVE SECOND ORDER OPTIMIZATION VIA FISHER INFORMATION|[Link](https://arxiv.org/abs/2405.16397)|
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|274|INNAprop|arXiv|2025|A SECOND-ORDER-LIKE OPTIMIZER WITH ADAPTIVE GRADIENT SCALING FOR DEEP LEARNING|[Link](https://arxiv.org/abs/2410.05871)|
|Second-Order Methods|Fisher Information Matrix Application|Diagonal Fisher Approximation|275|OCAR|CoRR|2025|Online Curvature-Aware Replay: Leveraging 2nd Order Information for Online Continual Learning|[Link](https://arxiv.org/abs/2502.01866)|
|Second-Order Methods|Hessian Approximation & Estimation|Block Hessian Approximation|276|Q-Newton|arXiv|2025|Q-Newton: Hybrid Quantum-Classical Scheduling for Accelerating Neural Network Training with Newton’s Gradient Descent|[Link](https://arxiv.org/abs/2405.00252)|
|Second-Order Methods|Fisher Information Matrix Application|Curvature-Aware Approximation|277|MAC|arXiv|2025|MAC: AN EFFICIENT GRADIENT PRECONDITIONING USING MEAN ACTIVATION APPROXIMATED CURVATURE|[Link](https://arxiv.org/abs/2506.08464)|
|Second-Order Methods|Quasi-Newton Methods|Stochastic BFGS|278|S-BFGS|arXiv|2025|EFFICIENT STOCHASTIC BFGS METHODS INSPIRED BY BAYESIAN PRINCIPLES|[Link](https://arxiv.org/abs/2507.07729)|
|Zeroth-Order Methods|Adaptive Methods|Projection-based Adaptive|279|ZO-SAH|arXiv|2025|Subspace-based Approximate Hessian Method for Zeroth-Order Optimization|[Link](https://arxiv.org/abs/2507.06125)|
|Second-Order Methods|Hessian Approximation & Estimation||280|Newton's Method|ANL|1982|Newton's method||
|Second-Order Methods|Hessian Approximation & Estimation||281|Gauss-Newton Method|Biometrika|1974|Quasi-Likelihood Functions, Generalized Linear Models, and the Gauss-Newton Method||
|Second-Order Methods|Quasi-Newton Methods||282|BFGS|SIAM Journal on scientific computing|1995|A limited memory algorithm for bound constrained optimization||
|Second-Order Methods|Fisher Information Matrix Application||283|Natural Gradient|Neural computation|1998|Natural gradient works efficiently in learning||
|Zeroth-Order Methods|Perturbation Optimization|Paired Perturbation Sampling|284|SPSA|Cat|2001|Global random optimization by simultaneous perturbation stochastic approximation||
|Zeroth-Order Methods|Adaptive Methods|Momentum-based Adaptive|285|ZO-AdaMM|NeurIPS|2019|Zeroth-Order Adaptive Momentum Method for Black-Box Optimization||
|First-Order Methods|Momentum-Enhanced SGD|Accelerated Momentum|286|SGDM|ICML|2013|On the importance of initialization and momentum in deep learning||
|First-Order Methods|Adaptive Learning Rate Methods||287|Adam|ICLR|2015|ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION||
|Second-Order Methods|Quasi-Newton Methods|Stochastic BFGS|288|L-BFGS|Mathematics of computation|1980|Updating quasi-newton matrices with limited storage||
|First-Order Methods|Hybrid Methods|Projection Gradient Hybrid|289|HVAdam|AAAI|2025|HVAdam: A Full-Dimension Adaptive Optimizer||
|First-Order Methods|Adaptive Learning Rate Methods|Second-Order Moment Adaptation|290|AdamNX|arXiv|2025|AdamNX: An Adam improvement algorithm based on a novel exponential decay mechanism for the second-order moment estimate||
|First-Order Methods|Adaptive Learning Rate Methods|Momentum-based Adaptive|291|ROOT|arXiv|2025|ROOT: Robust Orthogonalized Optimizer for Neural Network Training||
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|292|DP-AdamW-BC|ICML|2025|DP-AdamW: Investigating Decoupled Weight Decay and Bias Correction in Private Deep Learning||
|Distributed Optimization Methods|Federated Learning Optimization|Federated Momentum Fusion|293|FedMuon|arXiv|2025|FedMuon: Accelerating Federated Learning with Matrix Orthogonalization||
|First-Order Methods|Gradient Normalization & Clipping|Layer-Wise Gradient Normalization|294|AuON|arXiv|2025|AuON: A Linear-time Alternative to Orthogonal Momentum Updates||
|Privacy-Preserving Optimization Methods|Differential Privacy Optimization|DP-SGD Variants|295|DP-MicroAdam|arXiv|2025|DP-MicroAdam: Private and Frugal Algorithm for Training and Fine-tuning||
|First-Order Methods|Accelerating Convergence Rate|Momentum Damping Mechanism|296|FANoS|arXiv|2026|FANoS: Friction-Adaptive Nos´e–Hoover Symplectic Momentum for Stiff Objectives||
|First-Order Methods|Hybrid Methods|Gradient Smoothing Hybrid|297|NOVAK|arXiv|2026|NOVAK: Unified adaptive optimizer for deep neural networks||
## ⚙️Usage
Please install the required dependencies:
```bash
pip install -r requirements.txt
```
## 📈Benchmarking
We provide a script for easy benchmarking. See [examples/benchmark](https://github.com/facebookresearch/hiera/blob/main/examples/benchmark.ipynb) to see how to use it.
## 🔗Citation

If you find our survey and repository useful for your research project, please consider citing our paper:

```bibtex
@misc{zhang2026optimsurvey,
      title={A Systematic Survey of Optimization Methods: Algorithms, Scenarios, and Evaluations},
      author={Tong Zhang, Jiangning Zhang},
      year={2026},
      publisher={GitHub},
      howpublished={\url{https://github.com/JZhangTon/optim_survey}}
}
```
## 📫Contact

```
22560294@zju.edu.cn
```

[![Star History Chart](https://api.star-history.com/svg?repos=JZhangTon/optim_survey&type=Date)](https://star-history.com/#JZhangTon/optim_survey&Date)











