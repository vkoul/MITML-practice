# MITML Practice Projects

Scaffolded Jupyter notebook projects for MIT 6.86x (Machine Learning with Python — From Linear Models to Deep Learning). Each notebook provides the data, boilerplate, and visualization — you implement the ML core.

## Projects

| # | Project | Unit | Key Concept |
|---|---------|------|-------------|
| 01 | [Perceptron](01-perceptron/) | 1 | Linear classification, decision boundaries |
| 02 | [SVM / Hinge Loss](02-svm-hinge-loss/) | 1 | Margins, hinge loss, regularization |
| 03 | [Linear Regression](03-linear-regression/) | 2 | Normal equation, gradient descent, ridge |
| 04 | [Kernel Methods](04-kernels/) | 2 | Kernel trick, RBF, polynomial kernel |
| 05 | [Neural Network](05-neural-network/) | 3 | Forward pass, backprop, ReLU, SGD |
| 06 | [RNN](06-rnn/) | 3 | Hidden states, BPTT, text generation |
| 07 | [CNN](07-cnn/) | 3 | Convolution, pooling, feature maps |
| 08 | [Capstone](08-capstone/) | All | End-to-end pipeline, model comparison |
| 09 | [Collaborative Filtering](09-collaborative-filtering/) | 2 | Matrix factorization, alternating minimization |
| 10 | [Clustering & EM](10-clustering-em/) | 4 | K-Means, GMM, E-step/M-step |
| 11 | [Reinforcement Learning](11-reinforcement-learning/) | 5 | Value iteration, Q-learning, MDPs |

## How It Works

Every notebook follows the same pattern:

1. **Data** — synthetic datasets, generated for you
2. **Algorithm** — `### YOUR CODE ###` blocks where you implement the core ML
3. **Sanity checks** — assert-based tests that validate your implementation
4. **Experiments** — guided explorations with pre-built visualization
5. **Reflection** — conceptual questions connecting implementation to theory

No ML frameworks. NumPy + matplotlib only. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the full design philosophy and [STRATEGY.md](STRATEGY.md) for the rationale behind project selection.

## Getting Started

```bash
pip install numpy matplotlib jupyter
jupyter notebook
```

Open any project folder and start with the notebook inside.

## Disclaimer

This repository is an **independent study aid** and is not affiliated with, endorsed by, or officially connected to MIT, MITx, or edX. The course content, structure, and materials for MIT 6.86x are the intellectual property of their respective creators. These practice projects were created to supplement personal learning and do not contain or reproduce any copyrighted course materials, assignments, or solutions.

## License

MIT License — see [LICENSE](LICENSE) for details.
