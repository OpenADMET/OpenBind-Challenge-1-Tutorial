"""Configuration file for the OpenBind blind challenge 1."""

# Structure dataset
STRUCTURE_METRICS = ["LDDT-PLI", "BiSyRMSD", "LDDT-LP"]
# Penalty applied to BiSyRMSD when OST cannot match the ligand (lower is better,
# so a large value is used; 20 Å is well outside any reasonable binding-site RMSD)
BISYRMSD_NAN_PENALTY: float = 20.0
