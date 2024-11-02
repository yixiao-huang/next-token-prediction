
# Mechanics of Next Token Prediction with Self-Attention
Code for [Mechanics of Next Token Prediction with Self-Attention](https://arxiv.org/abs/2403.08081). Yingcong Li*, Yixiao Huang*, M. Emrullah Ildiz, Ankit Singh Rawat, and Samet Oymak
International Conference on Artificial Intelligence and Statistics (AISTATS), 2024

### Abstract
Transformer-based language models are trained on large datasets to predict the next token given an input sequence. Despite this simple training objective, they have led to revolutionary advances in natural language processing. Underlying this success is the self-attention mechanism. In this work, we ask: *What does a single self-attention layer learn from next-token prediction?* 

We show that training self-attention with gradient descent learns an automaton which generates the next token in two distinct steps: 

**(1) Hard retrieval:** Given the input sequence, self-attention precisely selects the *high-priority input tokens* associated with the last input token. 

**(2) Soft composition:** It then creates a convex combination of the high-priority tokens from which the next token can be sampled. 

Under suitable conditions, we rigorously characterize these mechanics through a directed graph over tokens extracted from the training data. We prove that gradient descent implicitly discovers the strongly-connected components (SCC) of this graph, and self-attention learns to retrieve the tokens that belong to the highest-priority SCC available in the context window. 

Our theory relies on decomposing the model weights into a directional component and a finite component that correspond to hard retrieval and soft composition steps, respectively. This also formalizes a related implicit bias formula conjectured in [Tarzanagh et al.~2023]. We hope that these findings shed light on how self-attention processes sequential data and pave the path toward demystifying more complex architectures.

### Requirements

```
cvxpy==1.4.1
matplotlib==3.4.1
numpy==1.20.2
scikit_learn==0.24.1
torch==1.8.1
```

### Paper link 
You can check more details at https://arxiv.org/abs/2403.08081
