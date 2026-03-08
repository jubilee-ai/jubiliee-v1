---
name: sklearn_unsupervised
description: Train a scikit-learn unsupervised learning estimator. Use this guide to select the right model for clustering, anomaly detection, or dimensionality reduction.
---

# Sklearn Generic - Unsupervised Model Selection

This skill trains scikit-learn **unsupervised learning** estimators. Use this guide to choose the correct model when there is **no target label** and the goal is to discover structure, segment records, detect anomalies, or compress features.

The scikit-learn unsupervised learning guide spans several major families: **Gaussian mixture models, clustering, matrix factorization / decomposition, covariance estimation, novelty and outlier detection, density estimation, and unsupervised neural-network-style methods**. This skill is designed to help route across that landscape while keeping a practical implementation focus on the most common tabular workflows. See the [scikit-learn unsupervised learning guide](https://scikit-learn.org/stable/unsupervised_learning.html) for the full taxonomy.

## How to Choose a Model

Work through these questions in order:

1. **What is the primary goal?** Clustering, anomaly detection, or dimensionality reduction?
2. **Do you know the number of groups ahead of time?** Some methods require `n_clusters`; others infer structure automatically.
3. **What shape are the groups?** Round/compact clusters behave differently from irregular-density clusters.
4. **How large is the dataset?** Some methods scale much better than others.
5. **Do you need interpretability or probabilistic outputs?** For example, soft cluster memberships vs. hard assignments.

Use the decision flowcharts below to narrow down the estimator, then confirm with the detailed tables.

---

## Clustering

### Quick Decision Flowchart

```
Start
 │
 ├─ Need fast, simple segmentation with known k? ──► KMeans
 │
 ├─ Same as KMeans but very large dataset? ──► MiniBatchKMeans
 │
 ├─ Need irregular cluster shapes or explicit noise handling?
 │    └─ DBSCAN
 │
 ├─ Need hierarchy / dendrogram-style grouping?
 │    └─ AgglomerativeClustering
 │
 ├─ Need soft probabilities or overlapping Gaussian-like groups?
 │    └─ GaussianMixture
 │
 └─ Not sure? ──► Start with KMeans, then compare with DBSCAN or GaussianMixture
```

### Clustering Reference

| Family | Estimator | When to Use |
|--------|-----------|-------------|
| **Centroid-based** | `KMeans` | Default clustering baseline; fast; works best for compact, roughly spherical clusters; requires `n_clusters` |
| | `MiniBatchKMeans` | Large datasets where standard KMeans is too slow; approximate but scalable |
| **Density-based** | `DBSCAN` | Unknown number of clusters; irregular cluster shapes; explicit noise/outlier labeling; sensitive to `eps`/`min_samples` |
| **Hierarchical** | `AgglomerativeClustering` | When you want nested grouping structure or hierarchical segmentation; useful when relationships matter more than centroids |
| **Probabilistic** | `GaussianMixture` | Soft cluster assignments; overlapping Gaussian-like segments; useful when membership probabilities matter |

---

## Gaussian Mixture Models

### Quick Decision Flowchart

```
Start
 │
 ├─ Need soft assignments instead of hard labels?
 │    └─ GaussianMixture
 │
 ├─ Believe each segment is approximately Gaussian / elliptical?
 │    └─ GaussianMixture
 │
 └─ Need a simpler, centroid-based baseline?
      └─ KMeans
```

### Gaussian Mixture Reference

| Family | Estimator | When to Use |
|--------|-----------|-------------|
| **Mixture model** | `GaussianMixture` | Model clusters as weighted Gaussian components; useful for soft segmentation, overlapping groups, and likelihood-based comparison via AIC/BIC |

---

## Anomaly Detection

### Quick Decision Flowchart

```
Start
 │
 ├─ Need a strong general-purpose anomaly baseline?
 │    └─ IsolationForest
 │
 ├─ Want dense-cluster logic where noise points are anomalies?
 │    └─ DBSCAN
 │
 └─ Need latent compression before downstream inspection?
      └─ PCA
```

### Anomaly Detection Reference

| Family | Estimator | When to Use |
|--------|-----------|-------------|
| **Tree-based** | `IsolationForest` | General-purpose outlier detection on tabular data; scales well; returns anomaly vs. normal assignments |
| **Density-based** | `DBSCAN` | Treat low-density points as noise/outliers while also clustering dense regions |
| **Projection-based** | `PCA` | Useful when anomalies are associated with poor low-dimensional reconstruction or variance structure |

---

## Dimensionality Reduction And Decomposition

### Quick Decision Flowchart

```
Start
 │
 ├─ Need fewer numeric features while preserving variance?
 │    └─ PCA
 │
 ├─ Need compact latent space before clustering?
 │    └─ PCA, then try KMeans or GaussianMixture
 │
 └─ Need interpretable original features instead of latent components?
      └─ Skip dimensionality reduction
```

### Dimensionality Reduction Reference

| Family | Estimator | When to Use |
|--------|-----------|-------------|
| **Linear projection** | `PCA` | Compress numeric information into fewer components; remove redundancy; useful before clustering or exploratory analysis |

---

## Broader Unsupervised Landscape

The scikit-learn guide also includes several unsupervised families beyond the core estimators currently wired into this skill runtime.

### Matrix Factorization And Component Methods

These methods are useful when the goal is latent representation learning rather than direct clustering:

| Family | Typical sklearn Methods | When to Use |
|--------|--------------------------|-------------|
| **Linear decomposition** | `PCA`, `TruncatedSVD` | Compress correlated numeric features or sparse matrices into fewer dimensions |
| **Non-negative decomposition** | `NMF` | Parts-based latent structure when all inputs are non-negative |
| **Topic / latent structure** | `LatentDirichletAllocation` | Unsupervised topic discovery in count-based text data |
| **Independent components** | `FastICA` | Separate independent sources or signals |
| **Factor models** | `FactorAnalysis` | Model latent factors driving observed covariance |

### Covariance Estimation

These are usually selected when the goal is robust covariance structure rather than segmentation:

| Family | Typical sklearn Methods | When to Use |
|--------|--------------------------|-------------|
| **Empirical / shrunk covariance** | `EmpiricalCovariance`, `LedoitWolf`, `OAS` | Estimate stable covariance matrices, often for downstream anomaly detection or portfolio/risk work |
| **Robust covariance** | `MinCovDet`, related robust estimators | Handle outliers while estimating covariance structure |

### Density Estimation

These methods model the data distribution directly:

| Family | Typical sklearn Methods | When to Use |
|--------|--------------------------|-------------|
| **Kernel density** | `KernelDensity` | Estimate smooth probability density for scoring, visualization, or anomaly heuristics |
| **Histogram / density views** | histogram-based analysis | Quick non-parametric understanding of feature distributions |

### Manifold Learning And Embeddings

These are mainly for exploration and visualization:

| Family | Typical sklearn Methods | When to Use |
|--------|--------------------------|-------------|
| **Neighborhood-preserving embeddings** | `Isomap`, `LLE`, `SpectralEmbedding`, `MDS`, `TSNE` | Visualize high-dimensional structure or derive low-dimensional embeddings |

### Unsupervised Neural Models

| Family | Typical sklearn Methods | When to Use |
|--------|--------------------------|-------------|
| **Energy-based models** | `BernoulliRBM` | Niche unsupervised representation learning, mostly for experimentation or historical workflows |

### Important Scope Note

This skill currently trains the following estimators directly:
- `KMeans`
- `MiniBatchKMeans`
- `DBSCAN`
- `AgglomerativeClustering`
- `GaussianMixture`
- `IsolationForest`
- `PCA`

The broader families above are included so the guide matches the scikit-learn unsupervised taxonomy, but not every estimator in those families is currently implemented in `train.py`.

---

## Model Selection Tips

- **Start with `KMeans` for clustering** unless there is a strong reason to prefer density-based or probabilistic structure.
- **Use `MiniBatchKMeans` for scale.** It is usually the safest first choice when row counts are large.
- **Use `DBSCAN` when cluster count is unknown** and you expect noise or non-spherical groups.
- **Use `GaussianMixture` when soft membership matters.** It is better than KMeans when you want probabilities instead of hard cluster assignments.
- **Use `IsolationForest` for anomaly detection** on general tabular datasets.
- **Use `PCA` for compression, not segmentation.** It reduces dimensionality but does not itself produce business-facing clusters.
- **Use AIC/BIC for Gaussian mixtures** when comparing component counts.
- **Be cautious with DBSCAN defaults.** Performance is highly sensitive to `eps` and `min_samples`.
- **Scale numeric features before distance-based methods.** KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture, and PCA are all sensitive to feature scale.
- **Treat dimensionality reduction and clustering as different goals.** PCA can improve a later clustering model, but it is not itself a clustering model.
- When in doubt, train 2-3 models from different families and compare silhouette score, Davies-Bouldin score, cluster counts, noise ratio, or explained variance.

## Example

```json
{
  "estimator": "KMeans",
  "model_name": "customer_segments_v1",
  "train_dataset_ref": "customer_features_train",
  "hyperparameters": {
    "n_clusters": 6,
    "random_state": 42
  }
}
```

## Workflow

After selecting an estimator using the guide above, follow these steps:

### Step 1 - Train

Call `train_with_skill(skill_name="unsupervised", params={...})` with:
- `"estimator"`: the sklearn class name (e.g. `"KMeans"`, `"DBSCAN"`, `"IsolationForest"`, `"PCA"`)
- `"model_name"`: a descriptive unique name (e.g. `"segments_v1"`, `"iforest_v2"`, `"pca_v1"`)
- `"train_dataset_ref"`: the training dataset reference
- `"feature_columns"`: (optional) explicit feature subset
- `"hyperparameters"`: (optional) override estimator hyperparameters

The training tool automatically preprocesses the data, fits the estimator, saves the artifact, and returns unsupervised diagnostics such as silhouette score, Davies-Bouldin score, inertia, anomaly ratio, or explained variance when applicable.

### Step 2 - Compare and iterate

Analyze the returned diagnostics, then either:
- Tune hyperparameters on the same estimator (for example, `n_clusters`, `eps`, `min_samples`, or `n_components`)
- Switch to a different estimator family when the current one does not match the data geometry

When satisfied, keep the best artifact for downstream use or exploratory analysis.