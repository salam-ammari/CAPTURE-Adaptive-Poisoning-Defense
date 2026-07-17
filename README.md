# CAPTURE

## Causal Attribution and Provenance-Guided Recovery against Adaptive Poisoning in Multi-Tenant Cloud Learning

This repository contains the research prototype and reproducibility code for the manuscript:

> **CAPTURE: Causal Attribution and Provenance-Guided Recovery against Adaptive Poisoning in Multi-Tenant Cloud Learning**

**Authors:** Salam Al-E’mari, Yousef Sanjalawe, and Muder Almiani  
**Year:** 2026  
**Research area:** Federated learning security, adaptive poisoning, causal attribution, provenance-guided recovery, concept drift, and cloud intrusion detection.

---

## Overview

Cloud-based intrusion-detection services increasingly use federated learning to train shared models across multiple tenants without centralizing raw network data. Although this architecture improves privacy, a compromised tenant can manipulate the global model through poisoned data, malicious labels, or crafted model updates.

This repository implements two main components:

- **CATO**: a collusive adaptive temporal-orchestration poisoning attack that distributes a targeted objective across tenants and training rounds while attempting to remain statistically plausible.
- **CAPTURE**: a defense framework that accumulates counterfactual influence over time, discounts evidence explained by legitimate drift, attributes suspicious coalitions, limits risky aggregation weights, and performs selective model recovery.

The implementation is an experimental research prototype designed to study the complete poisoning lifecycle:

1. attack generation;
2. federated aggregation;
3. temporal detection;
4. tenant attribution;
5. coalition identification;
6. drift-aware monitoring;
7. selective model recovery; and
8. empirical robustness analysis.

---

## Main Research Contributions

The code supports the following mechanisms described in the manuscript:

- adaptive poisoning through multiple CATO variants;
- exact counterfactual influence estimation on a clean probe set;
- temporally accumulated influence evidence;
- drift-aware sequential detection;
- per-tenant causal-risk scoring;
- temporal coalition attribution;
- risk-adjusted aggregation;
- provenance-guided minimum-removal recovery;
- periodic exact recomputation during recovery;
- comparison with FedAvg, coordinate-wise median, trimmed mean, and Krum;
- empirical evaluation of a conditional global-model deviation bound.

---

## Repository Structure

```text
CAPTURE-Adaptive-Poisoning-Defense/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── runner.py
├── sim_core.py
├── sim_defense.py
│
├── results/
│   ├── results_DATA.json
│   ├── results_E1.json
│   ├── results_E2.json
│   ├── results_E3.json
│   ├── results_E4.json
│   ├── results_E5.json
│   ├── results_E6.json
│   ├── results_E7.json
│   ├── results_E8.json
│   └── results_curves.json
│
├── figures/
│   └── generated figures
│
└── tests/
    ├── test_aggregators.py
    ├── test_capture.py
    └── test_recovery.py
```

### Core files

| File | Purpose |
|---|---|
| `runner.py` | Builds the federated environment, runs experiments E1–E8, calculates metrics, and writes JSON result files. |
| `sim_core.py` | Defines the synthetic corpus, tenant partitioning, neural network, local training, predictions, and evaluation metrics. |
| `sim_defense.py` | Implements CATO, robust aggregation baselines, CAPTURE detection, coalition attribution, and selective recovery. |

> `sim_core.py` is required. The program will not run if this module is missing.

---

## Experimental Setting

The default testbed follows the configuration described in the manuscript.

| Parameter | Default value |
|---|---:|
| Total records | 40,000 |
| Training records | 32,000 |
| Test records | 5,000 |
| Clean probe records | 3,000 |
| Features | 32 |
| Classes | 9 |
| Default tenants | 20 |
| Malicious tenant fraction | 30% |
| Dirichlet concentration | 0.3 |
| Federated rounds | 28 |
| Local epochs | 2 |
| Learning rate | 0.15 |
| Batch size | 128 |
| Target class | 3 |
| Victim class | 0, representing benign traffic |
| Global seed | Defined in `sim_core.py` |

The corpus is synthetically generated to reproduce several structural properties of multi-tenant intrusion-detection data:

- class imbalance;
- rare malware families;
- correlated standardized features;
- severe non-IID tenant partitions;
- multiple malicious tenants;
- targeted class-to-benign poisoning;
- gradual covariate drift; and
- temporally distributed attacks.

The current repository does **not** include real tenant traffic or packet captures.

---

## Model

The shared classifier is a compact multilayer perceptron implemented in `sim_core.py`.

The manuscript configuration is:

```text
Input layer:   32 features
Hidden layer:  64 ReLU units
Output layer:  9 classes
Loss:          cross-entropy
Optimizer:     local SGD
```

The compact architecture enables direct counterfactual evaluation and full-history replay without requiring approximation methods intended for very large models.

---

## CATO Attack Variants

CATO generates targeted poisoning updates that map the selected malware class to the benign class.

| Variant | Strategy |
|---|---|
| `CATO-S` | Slow single-source poisoning distributed across rounds. |
| `CATO-C` | Complementary collusion across multiple malicious tenants. |
| `CATO-A` | White-box adaptive attack with stronger conformity to honest updates. |
| `CATO-D` | Poisoning shaped to resemble legitimate gradual drift. |
| `CATO-X` | Cross-tenant transfer-oriented poisoning. |
| `CATO-M` | Heterogeneous malicious mechanisms pursuing one target. |
| `CATO-O` | Intermittent participation that thins the temporal evidence trail. |
| `Naive` | Scaled target-label flipping used as a non-adaptive baseline. |

The default E1 and E2 experiments evaluate:

```text
Naive
CATO-S
CATO-C
CATO-A
CATO-O
```

Additional profiles are implemented in `sim_defense.py` and can be added to the runner configuration.

---

## Defenses and Aggregation Baselines

The repository compares CAPTURE with four aggregation rules:

| Method | Description |
|---|---|
| `FedAvg` | Data-volume-weighted federated averaging. |
| `Median` | Coordinate-wise median aggregation. |
| `TrimmedMean` | Coordinate-wise trimmed mean with a default trimming ratio of 0.2. |
| `Krum` | Selects an update using pairwise distance-based Byzantine robustness. |
| `CAPTURE` | Risk-adjusted aggregation with temporal causal evidence and attribution. |
| `CAPTURE-Monitor` | Runs CAPTURE monitoring and attribution while retaining FedAvg-style aggregation, allowing recovery to be tested on a genuinely compromised trajectory. |

---

## CAPTURE Pipeline

For every participating tenant and training round, CAPTURE performs the following operations.

### 1. Counterfactual influence

CAPTURE estimates whether removing a tenant's weighted update decreases harmful behavior on the clean target-class probe.

A positive influence score indicates that the contribution increases the target-to-benign misclassification behavior.

### 2. Temporal evidence accumulation

Small per-round effects are accumulated over a sliding window with exponential decay. This allows the detector to identify poisoning that is deliberately distributed across time.

### 3. Drift discounting

The input-feature mean of each tenant is compared with its previous value. Evidence explained by legitimate covariate drift is discounted before updating the sequential evidence process.

### 4. Sequential flagging

Each tenant maintains an evidence statistic. A tenant is flagged when its accumulated evidence exceeds the configured threshold.

### 5. Coalition attribution

Flagged tenants are grouped according to overlap between their positive-influence rounds. This is intended to detect colluding tenants even when their update vectors are deliberately dissimilar.

### 6. Risk-adjusted aggregation

Tenant aggregation weights are exponentially reduced according to accumulated causal risk.

### 7. Selective recovery

After detection, CAPTURE restores a safe checkpoint, removes tenants in descending causal-risk order, and replays the surviving history until residual attack success falls below the recovery threshold.

Periodic exact local recomputation is used to reduce stale-update error.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/CAPTURE-Adaptive-Poisoning-Defense.git
cd CAPTURE-Adaptive-Poisoning-Defense
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A minimal dependency file should include:

```text
numpy>=2.0,<3.0
```

Add any additional packages used by `sim_core.py`, tests, or figure-generation scripts.

---

## Running the Experiments

The experiment runner accepts one or more stage names.

### Run every experiment

```bash
python runner.py
```

When no stage is supplied, the runner executes all registered stages.

### Run one experiment

```bash
python runner.py E1
```

### Run several experiments

```bash
python runner.py E1 E2 E3
```

### Generate dataset statistics

```bash
python runner.py DATA
```

### Generate representative learning curves

```bash
python runner.py curves
```

---

## Experiment Stages

### E1 — Attack × Defense Matrix

```bash
python runner.py E1
```

Compares:

```text
Defenses:
FedAvg, Median, TrimmedMean, Krum, CAPTURE

Attacks:
None, Naive, CATO-S, CATO-C, CATO-A, CATO-O
```

Reported metrics:

- final macro-F1;
- final attack success rate;
- mean across seeds; and
- approximate 95% confidence interval.

Output:

```text
results_E1.json
```

---

### E2 — Detection and Coalition Attribution

```bash
python runner.py E2
```

Evaluates CATO variants at malicious fractions of:

```text
10%, 20%, and 30%
```

Reported metrics:

- precision;
- recall;
- detection F1;
- AUROC;
- detection delay; and
- coalition Jaccard score.

Output:

```text
results_E2.json
```

---

### E3 — Drift Safety

```bash
python runner.py E3
```

Measures honest-tenant false flags:

- under gradual covariate drift; and
- without drift.

Output:

```text
results_E3.json
```

---

### E4 — Selective Recovery

```bash
python runner.py E4
```

Runs CAPTURE in monitor mode so that the global model becomes genuinely compromised, then applies selective recovery.

Reported values include:

- pre-recovery ASR;
- pre-recovery macro-F1;
- residual ASR;
- recovered macro-F1;
- clean-reference macro-F1;
- removed-tenant fraction;
- number of removal steps;
- exact recomputation count;
- recovery time;
- retraining time; and
- estimated speedup.

Output:

```text
results_E4.json
```

---

### E5 — Conditional Deviation Bound

```bash
python runner.py E5
```

Compares the empirical model deviation against the conditional upper bound derived in the paper.

Reported values include:

- per-round empirical deviation;
- per-round bound;
- surviving malicious aggregation mass;
- update norm bound;
- assumed smoothness constant;
- bound tightness; and
- final empirical-to-bound ratio.

Output:

```text
results_E5.json
```

---

### E6 — Scalability

```bash
python runner.py E6
```

Measures average server time per round for:

```text
10, 20, 50, and 100 tenants
```

CAPTURE is compared with FedAvg.

Output:

```text
results_E6.json
```

---

### E7 — Defense Comparison / Ablation Baseline

```bash
python runner.py E7
```

The current implementation compares:

- full CAPTURE;
- FedAvg without the CAPTURE risk mechanism;
- coordinate-wise median; and
- Krum.

Reported metrics:

- final ASR; and
- final macro-F1.

Output:

```text
results_E7.json
```

> This stage is best interpreted as a reduced defense comparison. A strict component-wise ablation would additionally disable CAPTURE's causal influence, drift adjustment, temporal accumulation, coalition module, and recovery components separately.

---

### E8 — Recovery Sensitivity

```bash
python runner.py E8
```

Studies the trade-off between recovery cost and fidelity by varying the exact-recomputation interval.

Default values:

```text
1, 2, 4, 8, and 999 rounds
```

Reported values include:

- residual ASR;
- recovered macro-F1;
- exact recomputation count;
- recovery time; and
- number of removed tenants.

Output:

```text
results_E8.json
```

---

### Dataset Summary

```bash
python runner.py DATA
```

Output:

```text
results_DATA.json
```

The file records:

- total sample count;
- feature count;
- class count;
- class distribution;
- benign fraction;
- target class;
- victim class; and
- number of federated rounds.

---

### Representative Curves

```bash
python runner.py curves
```

Produces per-round macro-F1 and ASR trajectories for:

```text
FedAvg
Krum
CAPTURE
```

under `CATO-C`.

Output:

```text
results_curves.json
```

---

## Result Files

By default, each experiment writes its output into the repository root using the following naming convention:

```text
results_<stage>.json
```

For a cleaner repository, move the files into a dedicated directory:

```text
results/
```

A figure-generation script can then read these JSON files to reproduce the manuscript's plots and tables.

---

## Evaluation Metrics

### Macro-F1

Macro-F1 gives equal importance to each class and is suitable for the long-tailed malware distribution used in the testbed.

### Attack Success Rate

Attack success rate measures the fraction of target-class samples classified as the attacker-selected victim class:

```text
target malware family  →  benign class
```

Lower values are better.

### Detection Metrics

The detector is evaluated using:

- precision;
- recall;
- F1;
- AUROC; and
- detection delay in federated rounds.

### Coalition Jaccard

Coalition attribution is compared with the true malicious tenant set using Jaccard similarity.

### Recovery Metrics

Recovery is evaluated using:

- residual ASR;
- recovered macro-F1;
- removed-tenant count;
- exact tenant recomputations;
- runtime; and
- speed relative to clean retraining.

---

## Reproducibility

The code uses deterministic pseudorandom-number generators and fixed experiment seeds.

The primary seeds are assigned within each experiment stage:

```text
E1:     100-series
E2:     200-series
E3:     300- and 350-series
E4:     400-series
E5:     500
E6:     600
E7:     700-series
Curves: 900
```

To improve statistical confidence, researchers may increase the number of seeds in `runner.py`.

For example, replace:

```python
for s in range(3):
```

with:

```python
for s in range(10):
```

Expect the execution time to increase accordingly.

---

## Important Implementation Notes

### Synthetic evaluation

The current implementation uses a controlled synthetic intrusion-detection testbed. Results should not be interpreted as direct evidence of performance on operational cloud traffic.

### Prototype provenance

The manuscript defines signed, append-only provenance tuples. The present simulation retains tenant identifiers, round numbers, update vectors, and aggregation weights required for replay, but it is not a production cryptographic provenance service.

A production implementation should add:

- tenant authentication;
- digital signatures;
- canonical update serialization;
- collision-resistant hashing;
- append-only storage;
- checkpoint signing;
- access control; and
- audit-log retention.

### Exact counterfactual probes

The testbed uses a compact model that permits direct counterfactual contribution evaluation. Large neural networks may require influence approximations, gradient-alignment methods, checkpoint tracing, or sampled exact replay.

### Honest false flags

CAPTURE should be used as an investigation and triage mechanism unless its operating threshold has been calibrated for the deployment. Automatic tenant suspension requires stricter false-positive controls.

### Recovery assumptions

Selective recovery assumes that:

- a trusted checkpoint exists;
- the relevant contribution history has been retained;
- tenant identities can be linked across rounds;
- surviving tenants can recompute local updates when required; and
- malicious tenants may continue attacking during recovery.

---

## Known Limitations

The research prototype has the following limitations:

1. Evaluation currently uses synthetic data.
2. The model is intentionally small.
3. The number of independent seeds is limited.
4. The current coalition rule primarily uses temporal overlap.
5. The prototype does not implement production cryptographic provenance.
6. The current E7 stage is not a complete component-wise ablation.
7. Stored update histories may require substantial storage for large models.
8. Exact counterfactual evaluation does not scale directly to foundation models.
9. CATO-O can delay detection by reducing participation frequency.
10. The framework requires a clean target-class probe for influence evaluation.

---

## Suggested Real-World Validation

Future evaluations should consider real federated intrusion-detection or IoT-security datasets while preserving native tenant or device partitions.

The validation protocol should report:

- class imbalance;
- client or tenant heterogeneity;
- temporal ordering;
- drift definition;
- attacker placement;
- target and victim classes;
- poisoning budget;
- attacker participation schedule;
- probe construction;
- communication cost;
- server runtime;
- update-history storage; and
- recovery recomputation cost.

---

## Recommended `requirements.txt`

A minimal starting point is:

```text
numpy>=2.0,<3.0
```

A broader research environment may include:

```text
numpy>=2.0,<3.0
scikit-learn>=1.5,<2.0
pandas>=2.2,<3.0
matplotlib>=3.9,<4.0
pytest>=8.0,<9.0
```

Only include packages actually imported by the final repository.

---

## Recommended `.gitignore`

```gitignore
__pycache__/
*.py[cod]
*.pyd
*.so

.venv/
venv/
env/

.vscode/
.idea/
.spyproject/
.spyderproject/

.DS_Store
Thumbs.db

*.log
*.tmp
*.bak

checkpoints/
models/
*.npy
*.npz
*.pkl

.env
.env.*
```

Do not ignore the small JSON result files used to reproduce the paper.

---

## Testing

Run the test suite with:

```bash
pytest -q
```

Recommended tests include:

- FedAvg weight normalization;
- median and trimmed-mean output shapes;
- Krum behavior for small tenant counts;
- deterministic environment creation;
- CATO update dimensionality;
- CAPTURE evidence updates;
- risk-weight normalization;
- coalition construction;
- recovery termination;
- residual ASR calculation; and
- reproducibility under fixed seeds.

---

## Citation

Until the manuscript receives final publication metadata, cite the repository and manuscript as follows.

### BibTeX

```bibtex
@article{alemari2026capture,
  author  = {Salam Al-E'mari and Yousef Sanjalawe and Muder Almiani},
  title   = {CAPTURE: Causal Attribution and Provenance-Guided Recovery against Adaptive Poisoning in Multi-Tenant Cloud Learning},
  journal = {Manuscript submitted for publication},
  year    = {2026}
}
```

After acceptance, replace the journal status with the final volume, issue, page range, DOI, and publication year.

### Repository citation

```bibtex
@software{capture_code_2026,
  author  = {Salam Al-E'mari and Yousef Sanjalawe and Muder Almiani},
  title   = {CAPTURE: Research Code for Adaptive Poisoning Attribution and Recovery},
  year    = {2026},
  url     = {https://github.com/YOUR-USERNAME/CAPTURE-Adaptive-Poisoning-Defense}
}
```

---

## License

Add a license file before making the repository public.

A common choice for research software is the **MIT License**. If the code is tied to institutional intellectual property, funding restrictions, or a patent application, confirm the appropriate license with the relevant institution before publication.

---

## Ethical Use

This repository includes implementations of adaptive poisoning attacks for controlled scientific evaluation.

Use the code only for:

- authorized research;
- defensive testing;
- academic reproducibility;
- security education; and
- evaluation in isolated environments.

Do not deploy the attack code against systems, tenants, datasets, or infrastructure without explicit authorization.

---

## Contact

For questions regarding the manuscript or implementation, open a GitHub issue or contact the corresponding author through the affiliation information provided in the paper.

---

## Acknowledgment

This repository is intended to support transparent evaluation and reproducibility of the CAPTURE framework. Researchers who extend the code are encouraged to report configuration changes, random seeds, additional baselines, and negative results.
