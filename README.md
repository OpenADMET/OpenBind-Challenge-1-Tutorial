# OpenBind Blind Challenge 1 Tutorial

A lightweight, CPU-friendly walkthrough of the workflow for the [OpenBind Blind Challenge 1](https://huggingface.co/spaces/openadmet/openbind-blind-challenge-1). The challenge asks participants to predict protein-ligand complex structures for a blinded set of ligands against the Zika virus NS2B-NS3 protease.

Running a full structure-prediction pipeline is out of scope for a notebook, so this tutorial focuses on data preparation, quality checks, and building a submission in the correct format.

## What's here

- `notebooks/structure_prediction.ipynb` — the tutorial notebook. It walks through:
  1. Loading the blinded ligand list from Hugging Face
  2. Preparing inputs (SMILES parsing, protein FASTA) for structure prediction
  3. Exploratory data analysis of the ligand set (scaffold diversity, descriptor distributions)
  4. Inspecting a pre-generated Boltz-2 protein-ligand complex
  5. Understanding how submissions are scored against reference structures
  6. Building and validating a submission zip
- `inputs/` — example inputs: the NS2B-NS3 FASTA sequence, a sample Boltz-2 YAML config, and a pre-generated complex (`blind_challenge_001_model_0.cif`)
- `outputs/` — an example, ready-to-submit structure zip (`example_structure_submission/`, 400 PDB files) plus a scratch directory for your own predictions
- `evaluation/` — the scoring code used to grade submissions (`LDDT-PLI`, `BiSyRMSD`, `LDDT-LP`, computed via OpenStructure and bootstrap-aggregated)
- `baseline-docs/` — co-folding reports for three baseline models run on the full 400-ligand set against the NS2B-NS3 protease: [AlphaFold 3](baseline-docs/ALPHAFOLD3_REPORT.md), [Boltz-2](baseline-docs/BOLTZ2_REPORT.md), and [OpenBind-0](baseline-docs/OPENBIND0_REPORT.md), covering engine/hardware setup, PoseBusters validity, and submission validation for each
- `validation/` — `validate_structure_submission`, which checks a submission zip before upload (correct PDB count, `LIG` residue naming, single-chain-count limits, ligand connectivity vs. the expected SMILES)

## Submission requirements

- A `.zip` archive containing exactly 400 PDB files
- One predicted complex per ligand, named `{id}.pdb` (e.g. `blind_challenge_001.pdb`)
- The ligand residue name in each file must be exactly `LIG`

Use `validation/structure_validation.py` to check your zip locally before uploading it to the [challenge portal](https://huggingface.co/spaces/openadmet/openbind-blind-challenge-1).

## Setup

```bash
conda env create -f environment.yaml
conda activate openbind_challenge_tutorial
jupyter lab notebooks/structure_prediction.ipynb
```

Optional: `nglview`, used for in-notebook 3D structure visualization, requires a Jupyter environment with widget support enabled.
