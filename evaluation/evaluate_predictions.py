"""Functions for evaluating the predictions of the OpenBind blind challenge 1."""

import numpy as np
import pandas as pd
from loguru import logger

from .config import (
    BISYRMSD_NAN_PENALTY,
    STRUCTURE_METRICS,
)

from .utils import bootstrap_sampling

# Residue names treated as solvent — excluded from ligand detection
_SOLVENT_RESIDUE_NAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD"})

# Sentinel returned when a per-compound structure score cannot be computed
_NAN_STRUCTURE_METRICS: dict[str, float] = {m: float("nan") for m in STRUCTURE_METRICS}

# ---------------------------------------------------------------------------
# Structure scoring
# ---------------------------------------------------------------------------

def score_single_structure(model_path: str, ref_path: str) -> dict[str, float]:
    """Score one predicted protein-ligand complex PDB against the reference.

    Runs SCRMSDScorer (BiSyRMSD + LDDT-LP) and LDDTPLIScorer (LDDT-PLI) and
    returns the top-scoring ligand pair ranked by LDDT-PLI then BiSyRMSD.
    Returns NaN for every metric if scoring fails for any reason, so that a
    single bad submission file does not abort the full evaluation.

    Args:
        model_path (str): Filesystem path to the predicted complex PDB file.
        ref_path (str): Filesystem path to the reference complex PDB file.

    Returns:
        dict[str, float]: Metric name → score. Keys: ``LDDT-PLI``, ``BiSyRMSD``,
            ``LDDT-LP``. Any metric that cannot be computed is ``np.nan``.

    """
    try:
        from ost.mol.alg.ligand_scoring import (  # type: ignore[import]
            LDDTPLIScorer,
            SCRMSDScorer,
        )
        from ost.mol.alg.scoring_base import PDBPrep  # type: ignore[import]

        model = PDBPrep(model_path, fault_tolerant=True)
        ref = PDBPrep(ref_path, fault_tolerant=True)

        model_lig = model.Select("rname=LIG")
        ref_lig = ref.Select("rname=LIG")

        logger.info("Scoring structure {} against reference {}", model_path, ref_path)

        scrmsd_sc = SCRMSDScorer(
            model=model,
            target=ref,
            model_ligands=[model_lig],
            target_ligands=[ref_lig],
        )
        lddt_pli_sc = LDDTPLIScorer(
            model=model,
            target=ref,
            model_ligands=[model_lig],
            target_ligands=[ref_lig],
        )

        results = []
        for i, j in scrmsd_sc.assignment:
            scrmsd_aux = scrmsd_sc.aux_matrix[i, j]
            pli_aux = lddt_pli_sc.aux_matrix[i, j]
            scrmsd_map = scrmsd_aux["chain_mapping"]
            pli_map = pli_aux["chain_mapping"]
            # SCRMSDScorer and LDDTPLIScorer may independently choose a
            # chain mapping that is the inverse of each other (e.g. A→B vs
            # B→A) for symmetric homo-oligomers — both are equally valid.
            # Accept either direction before raising a mismatch error.
            inverted = {v: k for k, v in scrmsd_map.items()}
            if scrmsd_map != pli_map and inverted != pli_map:
                raise ValueError(f"Chain mapping mismatch: {scrmsd_map} vs {pli_map}")
            results.append(
                {
                    "LDDT-PLI": lddt_pli_sc.score_matrix[i, j],
                    "BiSyRMSD": scrmsd_sc.score_matrix[i, j],
                    "LDDT-LP": scrmsd_aux["lddt_lp"],
                }
            )

        if not results:
            logger.warning(
                "No ligand assignment found between {} and {} — returning NaN scores.",
                model_path,
                ref_path,
            )
            return dict(_NAN_STRUCTURE_METRICS)

        results.sort(key=lambda r: (-r["LDDT-PLI"], r["BiSyRMSD"]))
        return results[0]

    except Exception as e:
        logger.exception("OST scoring failed for {} vs {}: {}", model_path, ref_path, e)
        return dict(_NAN_STRUCTURE_METRICS)


def score_structure_predictions(
    predicted_structures: dict[str, str],
    ground_truth_structures: dict[str, str],
) -> pd.DataFrame:
    """Score all predicted protein-ligand complex PDB files against ground truth.

    Iterates over ``predicted_structures``, skipping any molecule ID not found
    in ``ground_truth_structures``. A ``coverage`` column (1.0 = matched,
    0.0 = failed) is added before NaN values are replaced with worst-case
    penalties so every compound contributes to bootstrap aggregation:

    - LDDT-PLI, LDDT-LP (↑ better): NaN → 0.0
    - BiSyRMSD (↓ better): NaN → ``BISYRMSD_NAN_PENALTY``

    Args:
        predicted_structures (dict[str, str]): Mapping from molecule ID to
            filesystem path of the predicted PDB file.
        ground_truth_structures (dict[str, str]): Mapping from molecule ID
            to filesystem path of the reference PDB file.

    Returns:
        pd.DataFrame: Per-compound scores with columns:
            ``Molecule Name``, ``LDDT-PLI``, ``BiSyRMSD``, ``LDDT-LP``,
            ``coverage`` (1.0 if successfully matched, 0.0 otherwise).

    """
    logger.info(
        "Scoring {} structure predictions against ground truth",
        len(predicted_structures),
    )
    rows = []
    for mol_id, model_path in predicted_structures.items():
        if mol_id not in ground_truth_structures:
            logger.warning("No ground truth found for {}, skipping.", mol_id)
            continue
        scores = score_single_structure(model_path, ground_truth_structures[mol_id])
        rows.append({"Molecule Name": mol_id, **scores})

    per_compound_df = pd.DataFrame(rows)
    n_scored = int(per_compound_df["LDDT-PLI"].notna().sum())
    logger.info(
        "Successfully scored {}/{} structures.",
        n_scored,
        len(predicted_structures),
    )

    # Record coverage before filling so the per-compound parquet is transparent
    per_compound_df["coverage"] = per_compound_df["LDDT-PLI"].notna().astype(float)

    # Apply worst-case penalties so every compound contributes to the bootstrap mean
    per_compound_df["LDDT-PLI"] = per_compound_df["LDDT-PLI"].fillna(0.0)
    per_compound_df["LDDT-LP"] = per_compound_df["LDDT-LP"].fillna(0.0)
    per_compound_df["BiSyRMSD"] = per_compound_df["BiSyRMSD"].fillna(
        BISYRMSD_NAN_PENALTY
    )

    return per_compound_df


def bootstrap_structure_metrics(
    per_compound_df: pd.DataFrame,
    n_bootstrap_samples: int,
) -> pd.DataFrame:
    """Bootstrap aggregate structure metrics over the set of scored compounds.

    Each bootstrap iteration resamples the compounds (rows) with replacement
    and computes the mean of each metric. Failed structures are already filled
    with worst-case penalty values by ``score_structure_predictions``, so
    ``np.mean`` is used — every compound contributes to the aggregate.
    The returned DataFrame uses the same ``Sample`` / ``Endpoint`` schema as
    ``bootstrap_metrics``, so ``average_bootstrap_results_by_endpoint`` can be
    reused unchanged. Coverage is a scalar property of the whole submission and
    is not bootstrapped — it is added to ``averaged_df`` separately in
    ``score_structure_submission``.

    Args:
        per_compound_df (pd.DataFrame): Output of ``score_structure_predictions``
            — one row per compound, columns ``LDDT-PLI``, ``BiSyRMSD``,
            ``LDDT-LP``, ``coverage``.
        n_bootstrap_samples (int): Number of bootstrap iterations.

    Returns:
        pd.DataFrame: Bootstrap results with columns:
            ``Sample``, ``Endpoint``, ``LDDT-PLI``, ``BiSyRMSD``, ``LDDT-LP``.

    """
    _BOOTSTRAP_COLS = STRUCTURE_METRICS
    scores = per_compound_df[_BOOTSTRAP_COLS].to_numpy()
    n_compounds = scores.shape[0]

    rows = []
    for sample_idx, idx in enumerate(
        bootstrap_sampling(n_compounds, n_bootstrap_samples)
    ):
        sample_means = np.mean(scores[idx], axis=0)
        row: dict[str, object] = {"Sample": sample_idx, "Endpoint": "Structure"}
        for col, value in zip(_BOOTSTRAP_COLS, sample_means):
            row[col] = value
        rows.append(row)

    return pd.DataFrame(rows)
