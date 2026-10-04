# Old runs: within-driver split (superseded)

Results from the first experiment round, kept for reference.

- Split: each of the 26 drivers kept a random stratified 80/20 train/test split of its own images.
  State Farm images are consecutive video frames, so test frames had near-duplicate neighbours in the
  training set; the baseline accuracies here (student 85.9%, teachers ~97%) are therefore optimistic.
- Contents: `baseline/` (FedKD, 26 clients, 30 rounds), `gi_bench/`, `gi_tune/`, `gi_s1_none/`,
  `gi_s2_dinov2/`, `gi_s3_vitb/` (GradInversion, 4k iterations), `S2_full/` (GradInversion S2, 20k
  iterations, with convergence plots), and `fedkd_gia_results.xlsx` (the workbook for all of the above).
- Reconstruction tensors (`*_rec.pt`) were never committed; image grids (PNG) are included.

Superseded by the driver-level split (5 unseen test drivers) in `../baseline/` and the reruns next to it.
