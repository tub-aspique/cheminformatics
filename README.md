# Cheminformatics & ML

Personal self-taught project exploring Machine Learning applied to chemistry,
built alongside my Master's degree in Chemistry (ENS Paris-Saclay).

## Objective

Explore how ML can be combined with chemical data and molecular descriptors
to predict physicochemical properties of molecules from their structure.
All models are trained on publicly available experimental datasets and evaluated
using rigorous cross-dataset validation protocols.

## Projects

### 01 — Aqueous Solubility Prediction
`notebooks/01_esol_solubility.ipynb`

Prediction of aqueous solubility (logS, mol/L) from molecular structure using
RDKit descriptors and Morgan fingerprints on the ESOL benchmark dataset [1].

**Training protocol:**
All models were trained using an 80/20 random train/test split (random_state=42
for reproducibility). No validation set was used for hyperparameter tuning in
this initial benchmark — default hyperparameters were applied throughout.
The 6 molecular features used are: LogP, molecular weight, number of H-bond
donors and acceptors, topological polar surface area (TPSA), and number of
aromatic rings. Morgan fingerprints (radius=2, 2048 bits) were added in a
second step and concatenated with the 6 descriptors to form a 2054-dimensional
feature vector.

**Internal benchmark (ESOL test set, n=226):**
| Model | R² | RMSE (log mol/L) |
|---|---|---|
| Linear Regression | 0.765 | 1.054 |
| Random Forest | 0.859 | 0.816 |
| RF + Morgan Fingerprints (r=2, 2048 bits) | 0.866 | 0.795 |
| XGBoost + Morgan Fingerprints | 0.879 | 0.757 |

**Cross-dataset validation (trained on AqSolDB\ESOL, tested on ESOL):**
| Training set | n (train) | Test set | n (test) | R² | RMSE |
|---|---|---|---|---|---|
| ESOL | 902 | ESOL | 226 | 0.879 | 0.757 |
| AqSolDB\ESOL | 9378 | ESOL | 226 | **0.906** | **0.668** |

**Discussion:**
The XGBoost model achieves R²=0.879 and RMSE=0.757 log mol/L on the ESOL
test set, consistent with published results on this benchmark: Delaney [1]
reported RMSE=0.89 with a linear model, and more recent deep learning
approaches achieve RMSE≈0.58 [5]. Our result sits between these two benchmarks,
which is expected given the limited feature set (6 physicochemical descriptors
+ 2D fingerprints) and absence of hyperparameter optimization.

The most predictive individual feature is LogP (lipophilicity), consistent
with the known negative correlation between hydrophobicity and aqueous
solubility. TPSA and H-bond donor/acceptor counts capture polar interactions
that promote solvation. The addition of Morgan fingerprints provides marginal
improvement (+0.7% R²), suggesting that the 6 physicochemical descriptors
already capture most of the relevant variance for this dataset.

Cross-dataset validation reveals that a model trained on the larger AqSolDB
dataset [2] (9378 molecules after removal of ESOL overlap) generalizes better
to external data (R²=0.906, RMSE=0.668), despite lower internal metrics.
This highlights a key principle: internal R² is insufficient to assess
true predictive power — external validation on leak-free datasets is essential.

Performance could be further improved by: (i) incorporating solvation energy
from DFT calculations (shown by Boobier et al. [3] to be the most predictive
descriptor for aqueous solubility), (ii) adding experimental melting point as
a proxy for lattice energy, and (iii) using graph neural networks (GNNs) to
learn structural features directly from molecular graphs rather than
hand-crafted descriptors.

---

### 02 — Organic Solvent Solubility Prediction
`notebooks/02_organic_solvent_solubility.ipynb`

Prediction of logS in three organic solvents (ethanol, benzene, acetone) using
the dataset from Boobier et al. [3]. Three feature sets compared: RDKit
descriptors + Morgan fingerprints, DFT-derived descriptors (14 descriptors
selected from B3LYP/6-31+G(d) calculations), and their combination.

**Training protocol:**
Each solvent was modelled independently. An 80/20 train/test split was applied
per solvent (random_state=42). The 14 DFT descriptors used are those retained
by Boobier et al. after correlation analysis: MW, MP (melting point), molar
volume, solvation free energy (ΔG_solv), dipole moment in solution (solv_dip),
orbital interaction energies (LsoluHsolv, LsolvHsolu), solvent-accessible
surface area (SASA), and partial charge descriptors (O_charges, C_charges,
Het_charges, Most_neg, Most_pos). DFT values were pre-computed at the
B3LYP/6-31+G(d) level with IEFPCM solvation (Gaussian 09 [4]) by Boobier
et al. and used directly from their published dataset.

**XGBoost results by solvent and feature set:**
| Feature set | Ethanol R² | Benzene R² | Acetone R² |
|---|---|---|---|
| RDKit + Morgan Fingerprints | 0.491 | 0.465 | 0.296 |
| DFT descriptors (Boobier et al.) | 0.547 | 0.620 | 0.476 |
| DFT + Morgan Fingerprints | 0.597 | 0.677 | 0.531 |

**Discussion:**
R² values for organic solvents (0.30–0.68) are substantially lower than for
aqueous solubility. Three factors explain this:

1. **Dataset size**: each solvent contains only 370–553 training molecules,
compared to >9000 for the aqueous model. Small datasets increase variance and
limit generalization.

2. **Experimental noise**: Boobier et al. [3] report that ethanol and acetone
data are particularly noisy due to water contamination and solvent volatility,
making R² and RMSE less reliable metrics in these cases. Their own ET models
achieve R²=0.50 (ethanol) and R²=0.42 (acetone), consistent with our results.

3. **Missing descriptors**: melting point is identified by Boobier et al. as
the single most important descriptor for organic solvent solubility, as it
reflects the lattice energy of the solid. It is included in the DFT feature
set here but was not available for the RDKit-only model. Its high importance
explains why the DFT model outperforms the RDKit model particularly in benzene
(+15% R²), where solute-solute interactions dominate.

The benzene model performs best (R²=0.677), consistent with [3], likely
because benzene interactions are dominated by well-captured van der Waals
forces. Performance could be improved by expanding the training set to
additional solvents and incorporating conformational averaging of DFT
descriptors.

---

### Streamlit App — Solubility Predictor
`app.py`

Interactive web application predicting aqueous and organic solvent solubility
from a SMILES string. Includes a Tanimoto-based applicability domain indicator
(Morgan fingerprints, radius=2, similarity computed against 1000 randomly
sampled training molecules) to flag predictions outside the training domain.
Water model trained on AqSolDB\ESOL (n=9378, XGBoost + Morgan fingerprints).

```bash
conda activate chemml
cd path/to/cheminformatics
python -m streamlit run app.py
```

---

## Tech Stack

- **RDKit** (2023.09) — molecular manipulation, descriptor calculation, fingerprints
- **scikit-learn** — Linear Regression, Random Forest
- **XGBoost** — gradient boosting regressor
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **Streamlit** — interactive web application

## References

[1] Delaney, J.S. ESOL: Estimating aqueous solubility directly from molecular
structure. *J. Chem. Inf. Comput. Sci.* **44**, 1000–1005 (2004).
https://doi.org/10.1021/ci034243x

[2] Sorkun, M.C., Khetan, A. & Er, S. AqSolDB, a curated reference set of
aqueous solubility and 2D descriptors for a diverse set of compounds.
*Sci. Data* **6**, 143 (2019). https://doi.org/10.1038/s41597-019-0151-1

[3] Boobier, S., Hose, D.R.J., Blacker, A.J. & Nguyen, B.N. Machine learning
with physicochemical relationships: solubility prediction in organic solvents
and water. *Nat. Commun.* **11**, 5753 (2020).
https://doi.org/10.1038/s41467-020-19594-z

[4] Frisch, M.J. et al. Gaussian 09, Revision D.03. Gaussian, Inc.,
Wallingford CT (2016).

[5] Lusci, A., Pollastri, G. & Baldi, P. Deep architectures and deep learning
in chemoinformatics: the prediction of aqueous solubility for drug-like
molecules. *J. Chem. Inf. Model.* **53**, 1563–1575 (2013).
https://doi.org/10.1021/ci400187y

## Reproduce the environment

```bash
conda create -n chemml python=3.10
conda activate chemml
conda install -c conda-forge rdkit
pip install scikit-learn pandas numpy matplotlib seaborn jupyter xgboost streamlit
```