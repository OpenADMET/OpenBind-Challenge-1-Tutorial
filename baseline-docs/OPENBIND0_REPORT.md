# OpenBind-0 (`v0.5.0`) Co-Folding Report — OpenBind Blind Challenge 1 (Zika Virus NS2B–NS3 Protease)

## 1. Executive Summary

All **400 blinded small-molecule ligands** (`blind_challenge_001` – `blind_challenge_400`) from the OpenBind Blind Challenge 1 dataset were co-folded with the heterodimeric **Zika virus NS2B–NS3 protease** using **OpenBind-0 v0.5.0 (`of3-ob`)** on an `NVIDIA GB10` GPU. Five diffusion samples were generated per ligand (`2,000` total complex structures). Evaluating all samples with **PoseBusters (`v0.6.5`)** and selecting the highest-confidence PoseBusters-passing sample per ligand achieved:

- **PoseBusters `dock` physical & geometric validity (18 checks)**: **400 / 400 (100.00%)** *(up from 365 / 400 [91.25%] at Rank-1 only)*
- **PoseBusters full 22-check validity (including tetrahedral chirality vs. SMILES)**: **377 / 400 (94.25%)** *(up from 324 / 400 [81.00%] at Rank-1 only)*
- **Official OpenBind Submission Validator (`validation/structure_validation.py`)**: **`✅ Submission is valid — ready to upload.`** (`0` errors across all 400 PDB structures in `openbind_zika_ns2b_ns3_of3_ob_submission.zip`)

---

## 2. Engine, Checkpoint, Hardware & Target Specification

- **Engine**: OpenBind-0 v0.5.0 (`of3-ob:latest` Docker container, Apache 2.0 License)
- **Model Checkpoint**: OpenBind-0 v0.5.0 (`of3-ob-2025-06-30-174k.pt`, training cutoff **`2025-06-30`**)
- **Compute Hardware**: `NVIDIA Grace Hopper / GB10` (128 GB unified memory)
- **Sampling Protocol**:
  - `--num_model_seeds=1`, `--num_diffusion_samples=5` (`2,000` total predicted 3D complexes across 400 ligands)
  - `--use_msa_server=True` (ColabFold MMseqs2 paired + unpaired MSA server)
  - `--use_templates=False` (template-free inference)
- **Target Construct** (`inputs/NS2B-NS3_protein_sequence.fasta`):
  - **Chain A (NS2B cofactor, 46 aa)**:
    `SMGKSVDMYIERAGDITWEKDAEVTGNSPRLDVALDESGDFSLVEE`
  - **Chain B (NS3 protease domain, 168 aa)**:
    `MKEVKKGETTDGVYRVMTRRLLGSTQVGVGVMQEGVFHTMWHVTKGAALRSGEGRLDPYWGDVKQDLVSYCGPWKLDAAWDGLSEVQLLAVPPGERAKNIQTLPGIFKTKDGDIGAVALDYPAGTSGSPILDKCGRVIGLYGNGVVIKNGSYVSAITQGKREEETPVE`
  - **Chain C (Small-molecule ligand, residue name `LIG`)**: 400 blinded SMILES from `openbind-blind-challenge-1-TEST-BLINDED.csv` (`9` to `36` heavy atoms, mean `22.37 ± 5.78` heavy atoms)

### Execution Command

```bash
docker run --rm --name zika_of3_ob --gpus all --ipc=host --shm-size=64g \
  -v $(pwd):/work \
  -v ~/.openfold3:/weights \
  -e OPENFOLD_CACHE=/weights \
  of3-ob:latest \
  run_openfold predict \
    --query_json=/work/inputs/of3_queries.json \
    --output_dir=/work/preds/of3_ob \
    --inference_ckpt_path=/weights/of3-ob-2025-06-30-174k.pt \
    --num_diffusion_samples=5 \
    --num_model_seeds=1 \
    --use_msa_server=True \
    --use_templates=False
```

---

## 3. PoseBusters (`v0.6.5`) Evaluation & Multi-Sample Selection ($N = 400$)

For each of the 400 ligands, all 5 diffusion samples (`sample_1` – `sample_5`) were ranked by `sample_ranking_score` and evaluated with PoseBusters (`redock` configuration without RMSD, requiring explicit bond-order assignment and stereochemical verification against the input SMILES via `select_best_posebusters_sample.py`). The highest-confidence sample passing all PoseBusters checks was selected for the final submission.

### Summary Pass Rates (`Rank-1` $\rightarrow$ `Selected`)

| Evaluation Tier | Rank-1 Sample Only | After 5-Sample Selection | Improvement |
| :--- | :---: | :---: | :---: |
| **Pass ALL 18 `dock` checks** (chemistry, 3D geometry, flatness, internal energy, protein distance & volume overlap) | 365 / 400 (**91.25%**) | **400 / 400 (100.00%)** | **+35 ligands (+8.75%)** |
| **Pass ALL 22 PoseBusters checks** (`dock` + formula, bonds, double-bond stereo & tetrahedral chirality vs. SMILES) | 324 / 400 (**81.00%**) | **377 / 400 (94.25%)** | **+53 ligands (+13.25%)** |

### Per-Check Breakdown ($N = 400$, `posebusters_results.csv`)

| PoseBusters Check | Category | Rank-1 Pass Rate | Selected Pass Rate | Failed Count (Selected) |
| :--- | :--- | :---: | :---: | :---: |
| `mol_pred_loaded` | Loading | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `mol_cond_loaded` | Loading | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `sanitization` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `inchi_convertible` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `all_atoms_connected` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `no_radicals` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `molecular_formula` | Identity vs. SMILES | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `molecular_bonds` | Identity vs. SMILES | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `double_bond_stereochemistry` | Stereochemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `tetrahedral_chirality` | Stereochemistry | 355 / 400 (88.75%) | **377 / 400 (94.25%)** | 23 |
| `bond_lengths` | 3D Geometry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `bond_angles` | 3D Geometry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `internal_steric_clash` | 3D Geometry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `aromatic_ring_flatness` | Ring / Bond Flatness | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `non-aromatic_ring_non-flatness` | Ring / Bond Flatness | 399 / 400 (99.75%) | **400 / 400 (100.00%)** | 0 |
| `double_bond_flatness` | Ring / Bond Flatness | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `internal_energy` | Conformer Strain Energy | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `protein-ligand_maximum_distance` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `minimum_distance_to_protein` | Intermolecular Distance | 366 / 400 (91.50%) | **400 / 400 (100.00%)** | 0 |
| `minimum_distance_to_organic_cofactors` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `minimum_distance_to_inorganic_cofactors` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `minimum_distance_to_waters` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_protein` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_organic_cofactors` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_inorganic_cofactors` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_waters` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |

- **Remaining Failures after 5-Sample Selection (23 ligands total; 0 physical/clash failures)**:
  - `tetrahedral_chirality`: 23 ligands

### Selected Confidence Rank Distribution ($N = 400$)

| Selected Diffusion Sample Rank | Count | Percentage |
| :--- | ---: | ---: |
| **Rank 1** (highest `sample_ranking_score`) | 345 | 86.25% |
| **Rank 2** | 36 | 9.00% |
| **Rank 3** | 11 | 2.75% |
| **Rank 4** | 7 | 1.75% |
| **Rank 5** | 1 | 0.25% |

---

## 4. Confidence Metrics Summary ($N = 400$ Selected Structures)

Full per-ligand metrics are recorded in `of3_ob_confidence_summary.csv`:

| Metric | Mean $\pm$ SD | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| `sample_ranking_score` (selected sample) | $1.0443 \pm 0.0104$ | $1.0460$ | $1.0078$ | $1.0672$ |
| `sample_ranking_score_max` (across 5 samples) | $1.0453 \pm 0.0098$ | $1.0467$ | $1.0176$ | $1.0672$ |
| `sample_ranking_score_min` (across 5 samples) | $1.0265 \pm 0.0103$ | $1.0287$ | $0.9827$ | $1.0415$ |
| `iptm` (overall interface pTM) | $0.9110 \pm 0.0080$ | $0.9118$ | $0.8721$ | $0.9349$ |
| `ptm` (overall complex pTM) | $0.9250 \pm 0.0070$ | $0.9260$ | $0.8875$ | $0.9418$ |
| `avg_plddt` (all atoms, 0–100) | $90.06 \pm 0.45$ | $90.08$ | $88.02$ | $91.45$ |
| `gpde` (global predicted distance error, Å) | $0.3831 \pm 0.0515$ | $0.3618$ | $0.3299$ | $0.6635$ |
| `iptm_AB` (NS2B–NS3 protein–protein interface) | $0.9054 \pm 0.0056$ | $0.9061$ | $0.8900$ | $0.9221$ |
| `iptm_AC` (NS2B–Ligand interface) | $0.8323 \pm 0.1212$ | $0.8909$ | $0.3453$ | $0.9398$ |
| `iptm_BC` (NS3–Ligand interface) | $0.9232 \pm 0.0751$ | $0.9587$ | $0.5714$ | $0.9835$ |
| `has_clash` (internal clash flag) | $0.0000 \pm 0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ |

---

## 5. Official OpenBind Submission Validation (`structure_prediction.ipynb`)

Validated using the official OpenBind validation suite (`validate_submission.py` wrapping `validation/structure_validation.py` from `OpenADMET/OpenBind-Challenge-1-Tutorial`):

- **Archive**: `openbind_zika_ns2b_ns3_of3_ob_submission.zip` (`14.0 MB`, containing `submission/blind_challenge_001.pdb` – `blind_challenge_400.pdb`)
- **Validation Output**: `✅ Submission is valid — ready to upload.`
  - `400 / 400` expected PDB filenames present (`blind_challenge_001.pdb` – `blind_challenge_400.pdb`)
  - `400 / 400` single `LIG` residue on Chain C (`≤ 3` chains total: `A`, `B`, `C`)
  - `400 / 400` RDKit `AssignBondOrdersFromTemplate` topological isomorphism matches against the reference SMILES in `openbind-blind-challenge-1-TEST-BLINDED.csv`

---
