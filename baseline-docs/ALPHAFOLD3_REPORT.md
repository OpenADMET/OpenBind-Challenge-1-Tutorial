# AlphaFold 3 (`3.0.4`) Co-Folding Report — OpenBind Blind Challenge 1 (Zika Virus NS2B–NS3 Protease)

## 1. Executive Summary

All **400 blinded small-molecule ligands** (`blind_challenge_001` – `blind_challenge_400`) from the OpenBind Blind Challenge 1 dataset were co-folded with the heterodimeric **Zika virus NS2B–NS3 protease** using **Google DeepMind AlphaFold 3 (`3.0.4`)** on an `NVIDIA GB10` GPU. Five diffusion samples were generated per ligand (`2,000` total complex structures). Evaluating all 2,000 samples with **PoseBusters (`v0.6.5`)** and selecting the highest-confidence PoseBusters-passing sample per ligand achieved:

- **PoseBusters `dock` physical & geometric validity (18 checks)**: **397 / 400 (99.25%)** *(up from 369 / 400 [92.25%] at Rank-1 only — the highest Rank-1 physical validity among all three models)*
- **PoseBusters full 22-check validity (including tetrahedral chirality vs. SMILES)**: **361 / 400 (90.25%)** *(up from 327 / 400 [81.75%] at Rank-1 only)*
- **Official OpenBind Submission Validator (`validation/structure_validation.py`)**: **`✅ Submission is valid — ready to upload.`** (`0` errors across all 400 PDB structures in `openbind_zika_ns2b_ns3_af3_submission.zip`)

---

## 2. Engine, Checkpoint, Hardware & Target Specification

- **Engine**: Google DeepMind AlphaFold 3 (`3.0.4`, source code licensed under Apache 2.0)
- **Model Parameters**: Google DeepMind AlphaFold 3 Model Parameters (`af3.bin`, training cutoff **`2021-09-30`**) — **Non-Commercial Use Only** (Subject to Google DeepMind AlphaFold 3 Model Parameters Terms of Use and Output Terms of Use)
- **Compute Hardware**: `NVIDIA Grace Hopper / GB10` (128 GB unified memory)
- **MSA & Template Pipeline**:
  - Precomputed once via `run_alphafold.py --run_data_pipeline=true --norun_inference` (`jackhmmer` + `hmmsearch`) and injected into all 400 per-ligand JSON inputs via `make_af3_inputs.py`:
    - **Chain A (NS2B cofactor, 46 aa)**: `80` unpaired sequences, `3,898` paired sequences, `4` PDB templates
    - **Chain B (NS3 protease domain, 168 aa)**: `383` unpaired sequences, `11,411` paired sequences, `4` PDB templates
- **Sampling & Batching Optimization**:
  - `modelSeeds = [42]`, `5` diffusion samples per ligand (`seed-42_sample-0` .. `seed-42_sample-4`, `2,000` total complex structures across 400 ligands)
  - Executed in batch mode via `--input_dir=/work/af3_inputs` (`--norun_data_pipeline --run_inference=true --jax_compilation_cache_dir=/work/jax_cache`), compiling XLA kernels once for the `[256, 512]` token bucket (~77 s warmup) and inferring each subsequent 5-sample complex in **~14.2 seconds** (~1.6 hours total wall-clock time for all 400 complexes).
- **Target Construct** (`inputs/NS2B-NS3_protein_sequence.fasta`):
  - **Chain A (NS2B cofactor, 46 aa)**:
    `SMGKSVDMYIERAGDITWEKDAEVTGNSPRLDVALDESGDFSLVEE`
  - **Chain B (NS3 protease domain, 168 aa)**:
    `MKEVKKGETTDGVYRVMTRRLLGSTQVGVGVMQEGVFHTMWHVTKGAALRSGEGRLDPYWGDVKQDLVSYCGPWKLDAAWDGLSEVQLLAVPPGERAKNIQTLPGIFKTKDGDIGAVALDYPAGTSGSPILDKCGRVIGLYGNGVVIKNGSYVSAITQGKREEETPVE`
  - **Chain C (Small-molecule ligand, residue name `LIG`)**: 400 blinded SMILES from `openbind-blind-challenge-1-TEST-BLINDED.csv`

---

## 3. PoseBusters (`v0.6.5`) Evaluation & Multi-Sample Selection ($N = 400$)

All 2,000 AlphaFold 3 mmCIF outputs (`400 ligands × 5 samples`) were converted to OpenBind-compliant PDB files (Chain A = NS2B, Chain B = NS3, Chain C = `LIG` with `CONECT` records) and evaluated with PoseBusters (`v0.6.5`) via `process_af3_submission.py`.

### Summary Pass Rates (`Rank-1` $\rightarrow$ `Selected`)

| Evaluation Tier | Rank-1 Sample Only (`posebusters_results_af3_rank1.csv`) | After 5-Sample Selection (`posebusters_results_af3.csv`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Pass ALL 18 `dock` checks** (chemistry, 3D geometry, flatness, internal energy, protein distance & volume overlap) | **369 / 400 (92.25%)** | **397 / 400 (99.25%)** | **+28 ligands (+7.00%)** |
| **Pass ALL 22 PoseBusters checks** (`dock` + formula, bonds, double-bond stereo & tetrahedral chirality vs. SMILES) | **327 / 400 (81.75%)** | **361 / 400 (90.25%)** | **+34 ligands (+8.50%)** |

### Per-Check Breakdown ($N = 400$, `posebusters_results_af3.csv`)

| PoseBusters Check | Category | Rank-1 Pass Rate | Selected Pass Rate | Failed Count (Selected) |
| :--- | :--- | :---: | :---: | :---: |
| `mol_pred_loaded` | Loading | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `mol_cond_loaded` | Loading | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `sanitization` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `inchi_convertible` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `all_atoms_connected` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `no_radicals` | Chemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `molecular_formula` | Identity vs. SMILES | 399 / 400 (99.75%) | **400 / 400 (100.00%)** | 0 |
| `molecular_bonds` | Identity vs. SMILES | 399 / 400 (99.75%) | **400 / 400 (100.00%)** | 0 |
| `double_bond_stereochemistry` | Stereochemistry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `tetrahedral_chirality` | Stereochemistry | 351 / 400 (87.75%) | **364 / 400 (91.00%)** | 36 |
| `bond_lengths` | 3D Geometry | 399 / 400 (99.75%) | **400 / 400 (100.00%)** | 0 |
| `bond_angles` | 3D Geometry | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `internal_steric_clash` | 3D Geometry | 398 / 400 (99.50%) | **400 / 400 (100.00%)** | 0 |
| `aromatic_ring_flatness` | Ring / Bond Flatness | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `non-aromatic_ring_non-flatness` | Ring / Bond Flatness | 399 / 400 (99.75%) | **399 / 400 (99.75%)** | 1 |
| `double_bond_flatness` | Ring / Bond Flatness | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `internal_energy` | Conformer Strain Energy | 399 / 400 (99.75%) | **400 / 400 (100.00%)** | 0 |
| `protein-ligand_maximum_distance` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `minimum_distance_to_protein` | Intermolecular Distance | 373 / 400 (93.25%) | **398 / 400 (99.50%)** | 2 |
| `minimum_distance_to_organic_cofactors` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `minimum_distance_to_inorganic_cofactors` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `minimum_distance_to_waters` | Intermolecular Distance | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_protein` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_organic_cofactors` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_inorganic_cofactors` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |
| `volume_overlap_with_waters` | Intermolecular Volume | 400 / 400 (100.00%) | **400 / 400 (100.00%)** | 0 |

- **Remaining Failures after 5-Sample Selection (39 ligands total)**:
  - `minimum_distance_to_protein`: 2 ligands
  - `non-aromatic_ring_non-flatness`: 1 ligand
  - `tetrahedral_chirality`: 36 ligands

### Selected Confidence Rank Distribution ($N = 400$)

| Selected Diffusion Sample Rank | Count | Percentage |
| :--- | ---: | ---: |
| **Rank 1** (highest `ranking_score`) | 360 | 90.00% |
| **Rank 2** | 18 | 4.50% |
| **Rank 3** | 10 | 2.50% |
| **Rank 4** | 10 | 2.50% |
| **Rank 5** | 2 | 0.50% |

---

## 4. Confidence Metrics Summary ($N = 400$ Selected Structures)

Full per-ligand metrics are recorded in `af3_confidence_summary.csv`:

| Metric | Mean $\pm$ SD | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| `sample_ranking_score` (selected sample) | $1.0513 \pm 0.0147$ | $1.0565$ | $0.9950$ | $1.0701$ |
| `sample_ranking_score_max` (across 5 samples) | $1.0524 \pm 0.0132$ | $1.0570$ | $1.0069$ | $1.0701$ |
| `sample_ranking_score_min` (across 5 samples) | $1.0253 \pm 0.0129$ | $1.0277$ | $0.9840$ | $1.0490$ |
| `iptm` (overall interface pTM) | $0.9116 \pm 0.0158$ | $0.9200$ | $0.8600$ | $0.9300$ |
| `ptm` (overall complex pTM) | $0.9246 \pm 0.0091$ | $0.9300$ | $0.8900$ | $0.9300$ |
| `avg_plddt` (all atoms, 0–100) | $91.91 \pm 0.38$ | $91.97$ | $90.09$ | $92.86$ |
| `lig_plddt` (ligand atoms, 0–100) | $88.44 \pm 11.32$ | $93.46$ | $51.74$ | $98.51$ |
| `fraction_disordered` | $0.2789 \pm 0.0045$ | $0.2800$ | $0.2400$ | $0.2800$ |
| `iptm_AB` (NS2B–NS3 protein–protein interface) | $0.9038 \pm 0.0075$ | $0.9000$ | $0.8900$ | $0.9200$ |
| `iptm_AC` (NS2B–Ligand interface) | $0.7434 \pm 0.1456$ | $0.8100$ | $0.3300$ | $0.8900$ |
| `iptm_BC` (NS3–Ligand interface) | $0.8787 \pm 0.1066$ | $0.9300$ | $0.5100$ | $0.9700$ |
| `has_clash` (internal clash flag) | $0.0000 \pm 0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ |

---

## 5. Official OpenBind Submission Validation (`structure_prediction.ipynb`)

Validated using the official OpenBind validation suite (`validate_submission.py` wrapping `validation/structure_validation.py` from `OpenADMET/OpenBind-Challenge-1-Tutorial`):

- **Archive**: `openbind_zika_ns2b_ns3_af3_submission.zip` (`13.9 MB`, containing `submission_af3/blind_challenge_001.pdb` – `blind_challenge_400.pdb`)
- **Validation Output**: `✅ Submission is valid — ready to upload.`
  - `400 / 400` expected PDB filenames present (`blind_challenge_001.pdb` – `blind_challenge_400.pdb`)
  - `400 / 400` single `LIG` residue on Chain C (`≤ 3` chains total: `A`, `B`, `C`)
  - `400 / 400` RDKit `AssignBondOrdersFromTemplate` topological isomorphism matches against the reference SMILES in `openbind-blind-challenge-1-TEST-BLINDED.csv`

---
