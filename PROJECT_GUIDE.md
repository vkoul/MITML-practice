# MIT 6.86x Practice Projects — Complete Guide

## Design Philosophy

### How projects were chosen

**Core principle: "Swap the engine, keep the chassis."**

Every project follows the same structure — data generation, boilerplate, visualization, sanity checks — but changes only the ML core. This means you spend your time implementing algorithms, not wrestling with setup code. The scaffolding stays familiar; the math is the new part.

**Why this works:**

1. **Projects-first learning.** You build the algorithm before fully understanding the theory. The notebook gives you just enough math to implement, then the experiments reveal *why* the math works. Theory clicks deeper when you've already seen the behavior.

2. **One concept per project.** Each notebook targets exactly one new idea. Project 2 (SVM) doesn't also teach regularization and gradient descent and kernel tricks — it teaches hinge loss and margins, period. Regularization is there, but it's the experiment, not the exercise.

3. **Progressive reuse.** Your perceptron (P1) becomes the baseline you beat with SVM (P2). Your SVM's linear boundary motivates kernels (P4). Your kernel trick motivates "what if we *learned* the feature map?" — neural networks (P5). Each project answers a question the previous one raised.

4. **Difficulty through data, not code.** The implementation stays approachable (10-30 lines per exercise). Difficulty comes from harder datasets — linearly separable → moons → spirals → concentric circles → XOR → character sequences → images. You implement the same `fit/predict` pattern each time, but on progressively harder problems that force you to understand why the simpler model failed.

5. **Experiments over exercises.** After implementing, you *experiment*: vary hyperparameters, compare models, observe failure modes. This builds intuition that problem sets can't. Watching a perceptron fail on moons while your neural net solves it is worth more than proving convergence bounds.

### How exercises within each project were selected

Each notebook has three layers:

- **Core implementation** (must-do): The 3-5 functions marked `### YOUR CODE ###`. These implement the algorithm from the chapter. If you can fill these in, you understood the lecture.

- **Experiments** (should-do): Guided explorations with pre-built visualization code. Vary γ, compare λ values, try harder datasets. These build intuition about *why* the algorithm works and when it breaks.

- **Reflection questions** (think about): 3-4 questions that connect implementation to theory. No code required — these are for your own understanding.

The selection rule: **if the course tests it, you implement it.** Hinge loss gradient? You compute it. Backprop chain rule? You derive each step. RNN hidden state update? You write the loop. No concept is learned passively.

---

## Project Map

| # | Project | Chapter(s) | Key Concept | Dataset |
|---|---------|-----------|-------------|---------|
| 01 | Perceptron | Ch 1–2 | Linear classification, decision boundaries | 2D linearly separable |
| 02 | SVM / Hinge Loss | Ch 3 | Margins, hinge loss, regularization | 2D separable + non-separable |
| 03 | Linear Regression | Ch 4 | Normal equation, GD, ridge, bias-variance | 1D + polynomial |
| 04 | Kernel Methods | Ch 5 | Kernel trick, RBF, polynomial kernel | Circles, XOR |
| 05 | Neural Network | Ch 6–7 | Forward pass, backprop, ReLU, SGD | Moons, spirals |
| 06 | RNN | Ch 8–9 | Hidden states, BPTT, text generation | Character-level text |
| 07 | CNN | Ch 10 | Convolution, pooling, feature maps | 8×8 line images |
| 08 | Capstone | All | End-to-end pipeline, model comparison | MNIST digits (3 vs 8) |
| 09 | Collaborative Filtering | Unit 2 | Matrix factorization, alternating minimization | Synthetic ratings |
| 10 | Clustering & EM | Unit 4 | K-Means, GMM, E-step/M-step | Blobs + elliptical |
| 11 | Reinforcement Learning | Unit 5 | Value iteration, Q-learning, MDPs | 5x5 Gridworld |

---

## Project Details

### Project 01: Perceptron from Scratch
**File:** `01-perceptron/perceptron_from_scratch.ipynb`
**Chapters:** 1 & 2 — Introduction, Linear Classifiers

**You implement:**
- `predict_single(x)` — dot product + sign
- `predict(X)` — vectorized prediction
- `train_step(X, y)` — one pass through data with updates on mistakes

**Key ideas:**
- The perceptron learns a linear decision boundary by correcting mistakes
- Convergence is guaranteed only for linearly separable data
- Training order and initialization affect the final boundary

**Experiments:**
- Animated decision boundary evolution across epochs
- Non-separable data (what happens when perceptron can't converge?)
- Margin computation (bridge to SVMs)

**Stretch goals:** Averaged perceptron, convergence bound verification, 3D data

---

### Project 02: SVM / Hinge Loss
**File:** `02-svm-hinge-loss/svm_from_scratch.ipynb`
**Chapter:** 3 — Hinge Loss, Margin, Regularization

**You implement:**
- `hinge_loss(X, y)` — max(0, 1 - y·(w·x + b))
- `compute_gradients(X, y)` — subgradient of hinge + L2 penalty
- `train_step(X, y)` — gradient descent update
- `predict(X)` — sign of w·x + b

**Key ideas:**
- SVMs maximize the margin (distance to the nearest point)
- Hinge loss penalizes points inside the margin, not just misclassifications
- λ controls the tradeoff: small λ = wider margin, more violations allowed

**Experiments:**
- Vary λ from 0.001 to 1.0 — observe margin width change
- Side-by-side: perceptron boundary vs SVM boundary on the same data
- Margin lines (dashed) visualization

---

### Project 03: Linear Regression
**File:** `03-linear-regression/linear_regression_from_scratch.ipynb`
**Chapter:** 4 — Linear Regression

**You implement:**
- `fit_closed_form(X, y)` — normal equation: w = (X'X)⁻¹X'y
- `fit_ridge(X, y, λ)` — ridge: w = (X'X + λI)⁻¹X'y
- `fit_gradient_descent(X, y, lr, epochs)` — iterative MSE minimization
- `predict(X)` — Xw

**Key ideas:**
- Closed-form vs iterative: when to use each
- Ridge shrinks coefficients → prevents overfitting on polynomial features
- The bias-variance tradeoff: λ=0 overfits, λ=large underfits

**Experiments:**
- Degree-10 polynomial features with λ ∈ {0, 0.01, 0.1, 10}
- Train vs test MSE comparison
- GD convergence curve

---

### Project 04: Kernel Methods
**File:** `04-kernels/kernel_svm_from_scratch.ipynb`
**Chapter:** 5 — Nonlinear Classification, Kernels

**You implement:**
- `polynomial_kernel(X1, X2, degree, c)` — (X1·X2 + c)^d
- `rbf_kernel(X1, X2, γ)` — exp(-γ||x-x'||²)
- `KernelPerceptron.fit()` — dual perceptron with kernel trick

**Key ideas:**
- When data isn't linearly separable, map to higher dimensions
- The kernel trick avoids explicit feature computation
- RBF kernel = infinite-dimensional feature space

**Experiments:**
- Circles dataset + XOR dataset (both non-linearly separable)
- γ sweep: 0.1 (underfitting) → 20 (overfitting)
- Polynomial vs RBF kernel comparison on same data

---

### Project 05: Neural Network
**File:** `05-neural-network/neural_net_from_scratch.ipynb`
**Chapters:** 6 & 7 — Neural Networks, SGD, Backpropagation

**You implement:**
- `relu(z)`, `relu_derivative(z)`, `sigmoid(z)` — activation functions
- `TwoLayerNet.forward(X)` — z1→a1→z2→ŷ
- `TwoLayerNet.compute_loss(ŷ, y)` — binary cross-entropy
- `TwoLayerNet.backward(X, y, ŷ)` — full backprop chain rule, weight updates

**Key ideas:**
- Hidden layers create nonlinear decision boundaries
- Backprop is just the chain rule applied layer by layer
- He initialization prevents dead neurons
- More hidden units = more complex boundaries (but risk of overfitting)

**Experiments:**
- Hidden size sweep: 2, 4, 16, 64
- Moons dataset (easy) vs spiral dataset (hard — tests network limits)
- Loss curve visualization

---

### Project 06: RNN — Character-Level Language Model
**File:** `06-rnn/rnn_from_scratch.ipynb`
**Chapters:** 8 & 9 — RNNs, LSTMs, Sequence Models

**You implement:**
- `CharRNN.forward(inputs, h_prev)` — RNN cell: h_t = tanh(W_xh·x + W_hh·h + b)
- `CharRNN.loss(probs, targets)` — cross-entropy over sequence
- `CharRNN.backward(probs, hs, xs, targets)` — BPTT (backprop through time)

**Key ideas:**
- Hidden state carries information across time steps
- BPTT unrolls the RNN and applies backprop to the unrolled graph
- Gradient clipping prevents exploding gradients
- Temperature controls generation diversity

**Experiments:**
- Text generation at different temperatures (0.3 → 1.5)
- Hidden size comparison: 16, 64, 128
- Observe how generated text improves with training

**Provided for you:** Text sampling function, gradient clipping, softmax

---

### Project 07: CNN — Convolution & Pooling
**File:** `07-cnn/cnn_from_scratch.ipynb`
**Chapter:** 10 — Convolutional Neural Networks

**You implement:**
- `conv2d(image, kernel)` — 2D convolution (valid padding)
- `max_pool2d(image, pool_size)` — max pooling with stride
- `SimpleCNN.forward_one(image)` — Conv → ReLU → Pool → Flatten → Sigmoid

**Key ideas:**
- Convolution filters detect local patterns (edges, textures)
- Pooling provides translation invariance
- Weight sharing (same filter across the image) reduces parameters
- In real CNNs, filters are learned — here you see hand-crafted vs random

**Experiments:**
- Edge detection with horizontal/vertical Sobel-like filters
- Hand-crafted filters vs random filters on line classification
- Translation invariance demo: same filter responds regardless of position

---

### Project 08: Capstone — End-to-End ML Pipeline
**File:** `08-capstone/ml_pipeline_capstone.ipynb`
**Chapters:** All (1–10)

**You do:**
- Load MNIST digits (3 vs 8) — a real dataset
- Adapt models from Projects 1–7 to 64-dimensional input
- Compare: perceptron, SVM, kernel perceptron, neural network
- Write a mini-report with findings

**Key ideas:**
- Real data is messier than synthetic data
- Simple models can be surprisingly competitive
- Model comparison requires consistent train/test splits
- The "best" model depends on what you're optimizing for

**Stretch goals:** Multi-class (10 digits), confusion matrix, PCA visualization, learning curves

---

### Project 09: Collaborative Filtering
**File:** `09-collaborative-filtering/collab_filtering_from_scratch.ipynb`
**Unit:** 2 — Collaborative Filtering, Recommender Systems

**You implement:**
- `predict(i, j)` — dot product u_i · v_j
- `predict_all()` — full matrix U @ V.T
- `compute_loss(R, mask)` — MSE on observed entries + L2 regularization
- `update_user(i, R, mask)` — ridge regression fixing V
- `update_item(j, R, mask)` — ridge regression fixing U

**Key ideas:**
- Low-rank matrix factorization: R ≈ UV^T captures latent user/item factors
- Alternating minimization: fix one matrix, solve for the other via ridge regression
- Only observed entries contribute to the loss — missing entries are what we predict
- Regularization prevents overfitting when most entries are missing

**Experiments:**
- Rank selection (1, 2, 3, 5, 10) — train/test RMSE curves
- Regularization sweep — λ effect on overfitting
- Baseline comparison: global mean RMSE vs factorization RMSE

---

### Project 10: Clustering & EM Algorithm
**File:** `10-clustering-em/clustering_em_from_scratch.ipynb`
**Unit:** 4 — Unsupervised Learning, Clustering, Mixture Models

**You implement (K-Means):**
- `assign(X)` — assign each point to the nearest centroid
- `update(X)` — recompute centroids as cluster means
- `inertia(X)` — sum of squared distances to assigned centroids

**You implement (GMM):**
- `e_step(X)` — compute responsibilities (soft assignments)
- `m_step(X, r)` — update means, covariances, mixing weights
- `log_likelihood(X)` — log-likelihood of data under the model

**Key ideas:**
- K-Means: hard assignments, assumes spherical clusters
- GMM/EM: soft assignments via responsibilities, learns covariance shape
- EM is guaranteed to increase (or not decrease) the log-likelihood at every step
- K-Means is a special case of EM with hard assignments and spherical covariances

**Experiments:**
- Elbow method for choosing K
- Initialization sensitivity (multiple random seeds)
- Side-by-side K-Means vs GMM on elliptical clusters with confidence ellipses

---

### Project 11: Reinforcement Learning
**File:** `11-reinforcement-learning/rl_from_scratch.ipynb`
**Unit:** 5 — Reinforcement Learning, MDPs, Value Iteration

**You implement:**
- `value_iteration(env, gamma)` — Bellman optimality update until convergence
- Policy extraction — argmax over action-values
- `q_learning(env, n_episodes)` — model-free learning with epsilon-greedy exploration

**Key ideas:**
- MDP framework: states, actions, rewards, transitions, discount factor
- Value iteration is model-based (knows transition probabilities)
- Q-learning is model-free (learns from experience via TD updates)
- Epsilon-greedy balances exploration vs exploitation

**Experiments:**
- Deterministic vs stochastic transitions (how does policy change?)
- Discount factor sweep (gamma = 0.5, 0.9, 0.99, 1.0)
- Exploration strategies: no decay vs fast decay

**Provided for you:** GridWorld environment, visualization, epsilon decay schedule

---

## Progression & Dependencies

```
P1 Perceptron ──→ P2 SVM (adds margins + regularization)
                    │
                    ├──→ P3 Regression (same math, continuous output)
                    │        └──→ P9 Collab Filtering (matrix factorization via ridge)
                    │
                    └──→ P4 Kernels (makes P1 nonlinear)
                           │
                           └──→ P5 Neural Net (learns the feature map)
                                  │
                                  ├──→ P6 RNN (sequential neural net)
                                  │
                                  └──→ P7 CNN (spatial neural net)

P10 Clustering & EM (standalone — unsupervised learning)

P11 Reinforcement Learning (standalone — learning from rewards)

P8 Capstone (all models compete on a real dataset)
```

## Technical Notes

- **No frameworks:** All projects use only NumPy and matplotlib. This forces you to understand every line.
- **Labels:** Projects 1-4 use {-1, +1} labels (standard for perceptron/SVM). Projects 5-7 use {0, 1} (standard for sigmoid/cross-entropy). The capstone handles the conversion.
- **Data:** Each project generates its own synthetic data. No downloads needed except the capstone (sklearn's digits).
- **Difficulty:** Projects 1–3 are ~30 min each. Projects 4–5 are ~1 hour. Projects 6–7 are ~1–2 hours. Projects 9–11 are ~1 hour each. The capstone is open-ended.
- **Course coverage:** Projects 1–4 cover Units 1–2 (linear classifiers, nonlinear classification). Project 9 covers collaborative filtering (Unit 2). Projects 5–7 cover Unit 3 (neural networks). Project 10 covers Unit 4 (unsupervised learning). Project 11 covers Unit 5 (reinforcement learning).
