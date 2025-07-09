# 🚀 GPU-Accelerated SVM Solvers for Binary & Multiclass Classification

This project is the current progress of the research at EMC2 lab at Illinois Institute of Technology.  It focuses on implementing efficient SVM solvers for both binary and multiclass classification, exploring the advantages of dual formulations and modern optimization techniques. It includes versions of the standard binary SVM and advanced multiclass methods like Crammer-Singer and Weston-Watkins, optimized using dual decomposition and simplex projection techniques. GPU parallelisation is undergoing.

## 1. Highlights 🔍

### Binary SVM Solver
- **Soft Margin Formulation**: Includes slack variables and hinge loss.
- **Dual Formulation**: Optimized using coordinate descent with analytical updates.
- **GPU Readiness**: Violation-based updates lend themselves to CUDA parallelism.

### Multiclass SVM Solvers
1. **Crammer-Singer SVM**: Optimizes a single loss over all classes with one slack per sample.
2. **Weston-Watkins SVM**: Penalizes all incorrect classes per sample (more expressive).
3. **Blondel et al. (2014) Solver**: Uses dual block coordinate updates and solves subproblems using Euclidean projection onto the simplex for fast convergence.

## 2. Solver Architecture 🧠

- **Gradient & Violation Based Updates**: Per-sample gradient computation and violation-based sample filtering.
- **Simplex Projection**: Each dual subproblem is solved using either:
  - **Sort-based O(k log k) projection** Implemented currently using the simplex projection.
  - **Expected O(k) pivot projection** 
  - **Fixed-point bisection (for large k or hardware-specific optimization)**

## Credits 🙌
Guided by Professor [Yutong Wang](https://yutongwang.me/) at Illinois Insitute of Technology
