# Project Strategy & Rationale

A reference doc for updating or extending the practice projects for MIT 6.86x.

---

## Course-to-Project Mapping

| Course Unit | Topics | Projects |
|------------|--------|----------|
| Unit 1 — Linear Classifiers | Perceptron, hinge loss, margins, regularization | P01, P02 |
| Unit 2 — Nonlinear, Regression, Collab Filtering | Kernels, polynomial features, ridge regression, matrix factorization | P03, P04, P09 |
| Unit 3 — Neural Networks | Feedforward, backprop, RNNs, CNNs | P05, P06, P07 |
| Unit 4 — Unsupervised Learning | K-Means, GMM, EM algorithm | P10 |
| Unit 5 — Reinforcement Learning | MDPs, value iteration, Q-learning | P11 |
| All units | End-to-end pipeline, model comparison | P08 (Capstone) |

## Design Principles

### 1. "Swap the engine, keep the chassis"
Every notebook follows the same skeleton: imports, synthetic data generation, visualization helpers, `### YOUR CODE ###` exercises, sanity checks with asserts, experiments, reflection questions. The student only changes the ML core. This reduces cognitive load and focuses attention on the algorithm.

### 2. Projects-first learning
Students implement before fully understanding theory. The notebook gives just enough math to code the algorithm. The experiments section then reveals *why* the math works. Theory clicks after implementation.

### 3. One concept per project
Each notebook targets exactly one new idea. P02 teaches hinge loss and margins, not also regularization and gradient descent. Ancillary concepts appear in experiments, not exercises.

### 4. Progressive difficulty through data, not code
Implementation stays approachable (10-30 lines per exercise). Difficulty comes from harder datasets: linearly separable -> moons -> spirals -> concentric circles -> XOR -> character sequences -> images -> missing ratings -> elliptical clusters -> stochastic environments.

### 5. Experiments over exercises
After implementing, students experiment: vary hyperparameters, compare models, observe failure modes. This builds intuition that problem sets cannot.

## Scaffolding Pattern

Each notebook provides:

| Layer | What | Student effort |
|-------|------|---------------|
| **Data** | `make_*()` functions generating synthetic data | None — provided |
| **Helpers** | Activation functions, PDF, visualization, one-hot | None — provided |
| **Core** | 3-5 functions marked `### YOUR CODE ###` with `raise NotImplementedError` | Must implement |
| **Sanity checks** | Assert-based tests that validate correctness | Run after implementing |
| **Training loop** | `fit()` method calling the student's functions | None — provided |
| **Experiments** | Guided explorations with pre-built plotting | Modify parameters, observe |
| **Reflection** | 3-4 conceptual questions | Think/write |

The selection rule: **if the course tests it, you implement it.** No concept is learned passively.

## Label Conventions

- Projects 01-04: labels are {-1, +1} (standard for perceptron/SVM math)
- Projects 05-07: labels are {0, 1} (standard for sigmoid/cross-entropy)
- Projects 09-11: no binary labels (regression, unsupervised, RL)
- Project 08 (capstone): handles the conversion

## How to Add a New Project

1. **Identify the gap.** Cross-reference the MIT 6.86x syllabus with existing projects. If a topic is tested in the course but not covered by a project, it's a candidate.

2. **Choose the right grain.** One project = one concept. If a topic has two distinct algorithms (e.g., K-Means and GMM), they can share a notebook if they're closely related. Otherwise, split.

3. **Follow the skeleton:**
   ```
   # Project N: [Name] from Scratch
   ## 1. Data (provided)
   ## 2. Algorithm — Your Turn (### YOUR CODE ###)
   ### Sanity checks (asserts)
   ## 3. Train (provided outer loop)
   ## 4. Experiments (guided, with plots)
   ## 5. What You Learned (reflection questions)
   ```

4. **Write sanity checks first.** Before writing the exercise, decide what a correct implementation must satisfy. Each exercise should have 2-5 assert statements checking shapes, signs, convergence, and boundary conditions.

5. **Provide the hard parts.** Data generation, visualization, training loops, gradient clipping — anything that's infrastructure rather than ML. The student should spend time on the algorithm, not on matplotlib config.

6. **Add experiments that reveal intuition.** At least one experiment should show *why* the algorithm exists (e.g., K-Means failing on elliptical data motivates GMM). At least one should show a hyperparameter's effect (e.g., gamma sweep, epsilon decay).

7. **Update PROJECT_GUIDE.md.** Add the project to the table, the details section, the progression diagram, and the technical notes.

## How to Update an Existing Project

1. **Read the current notebook** and the corresponding course lecture/notes.
2. **Check:** Does every testable concept have an exercise? Does every exercise have sanity checks?
3. **Don't break the pattern.** The `### YOUR CODE ### / raise NotImplementedError` convention, the assert-based checks, and the provided helpers must stay consistent.
4. **Add experiments, not exercises.** If a concept is adjacent to the main topic (e.g., adding momentum to SGD in P05), make it an experiment, not a core exercise.

## Potential Future Additions

If the course adds or emphasizes new topics:

- **Bayesian methods** — could extend P03 (regression) with a Bayesian linear regression section
- **Dimensionality reduction (PCA)** — could be a standalone project or added to P10
- **Policy gradient methods** — could extend P11 with REINFORCE
- **Attention / Transformer** — if course covers it, would need a new project (P12)
- **Multi-class classification** — now a core exercise in P08 (One-vs-All); could be extended with other strategies (softmax regression, one-vs-one)

## Validation Checklist

Before shipping any update, verify:

- [ ] Every `### YOUR CODE ###` block has a matching `raise NotImplementedError`
- [ ] Every exercise has at least 2 assert-based sanity checks
- [ ] All notebooks parse as valid JSON (valid .ipynb)
- [ ] No imports beyond numpy, matplotlib (and scipy.linalg for edge cases)
- [ ] Data generation uses `seed=42` for reproducibility
- [ ] All experiments have pre-built visualization code
- [ ] Reflection questions connect implementation to theory
- [ ] PROJECT_GUIDE.md is updated with new/changed projects
