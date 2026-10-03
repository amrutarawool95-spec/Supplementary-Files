Supplementary File S2 - benchmark code and results
Inputs: tsuboyama_wt_40_72.csv (Dataset A, Tsuboyama et al. 2023), nielsen_2024_external.csv (Dataset B, Nielsen et al. 2024), curated subsets.
Run: pip install biopython scikit-learn scipy pandas matplotlib ; python analysis.py ; python rfigs.py
Outputs: cv_results.csv, leave_topology_out.csv, external_transfer.csv, descriptor tables.
Notes: sequence-only descriptors; TL column is labelled TL in the source data; length-only leave-topology-out is undefined (constant prediction).
