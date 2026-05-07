# Runtime Comparison: Two-stage vs DRL

- generated_at: 2026-04-25 23:07:56
- repeats: 1
- input: DTW matrices (.npy)
- drl_model: models/drl_delta_plus_batch_mean_s0.pth

| dataset | twostage mean±std (s) | drl mean±std (s) | twostage speedup (x) | drl speedup (x) | note |
|---|---:|---:|---:|---:|---|
| delta_cog__ais_filtered_2021-01-01_dtw_matrix | 0.0090 ± 0.0000 | 2.3355 ± 0.0000 | 258.7011 | 0.0039 |  |
| delta_cog__ais_filtered_2021-01-11_dtw_matrix | 0.0123 ± 0.0000 | 0.0167 ± 0.0000 | 1.3612 | 0.7347 |  |
| delta_cog__ais_filtered_2021-02-18_dtw_matrix | 0.0073 ± 0.0000 | 0.0139 ± 0.0000 | 1.9100 | 0.5236 |  |
| delta_cog__ais_filtered_2021-02-24_bbox_lat45_47.5_lon-130_-120_dtw_matrix | 0.0162 ± 0.0000 | 0.0167 ± 0.0000 | 1.0283 | 0.9725 |  |
| delta_cog__ais_filtered_2021-02-24_dtw_matrix | 0.0251 ± 0.0000 | 0.0178 ± 0.0000 | 0.7102 | 1.4081 |  |
| delta_cog__ais_filtered_2021-03-03_dtw_matrix | 0.0478 ± 0.0000 | 0.0224 ± 0.0000 | 0.4690 | 2.1323 |  |
| delta_cog__ais_filtered_2021-09-16_dtw_matrix | 0.0188 ± 0.0000 | 0.0166 ± 0.0000 | 0.8802 | 1.1361 |  |
| delta_cog__ais_sample2_dtw_matrix | 0.0278 ± 0.0000 | 0.0197 ± 0.0000 | 0.7103 | 1.4079 |  |
| delta_cog__ais_sample2_filtered_dtw_matrix | 0.0204 ± 0.0000 | 0.0167 ± 0.0000 | 0.8218 | 1.2169 |  |
| delta_cog__ais_sample2_pruned_k3_10_dtw_matrix | 0.0098 ± 0.0000 | 0.0147 ± 0.0000 | 1.5048 | 0.6646 |  |
| delta_cog__ais_sample2_segments_dtw_matrix | 0.0210 ± 0.0000 | 0.0166 ± 0.0000 | 0.7905 | 1.2650 |  |
| delta_cog__ais_sample2_segments_v2_dtw_matrix | 0.0139 ± 0.0000 | 0.0154 ± 0.0000 | 1.1108 | 0.9003 |  |
| delta_cog__ais_sample2_segments_v2_pruned_k3_10_dtw_matrix | 0.0120 ± 0.0000 | 0.0149 ± 0.0000 | 1.2390 | 0.8071 |  |
| delta_cog__ais_sample5_dtw_matrix | 0.0075 ± 0.0000 | 0.0132 ± 0.0000 | 1.7691 | 0.5652 |  |
| delta_cog__ais_yuanli_cleaned_dtw_matrix | 0.0086 ± 0.0000 | 0.0142 ± 0.0000 | 1.6509 | 0.6057 |  |
| delta_cog__ais_yuanli_dtw_matrix | 0.0148 ± 0.0000 | 0.0183 ± 0.0000 | 1.2361 | 0.8090 |  |
| delta_cog__clean_ais_filtered_dtw_matrix | 0.0158 ± 0.0000 | 0.0160 ± 0.0000 | 1.0110 | 0.9891 |  |
| delta_cog__final_dtw_matrix | 0.0105 ± 0.0000 | 0.0149 ± 0.0000 | 1.4208 | 0.7038 |  |
| delta_cog__iter_00_dtw_matrix | 0.0080 ± 0.0000 | 0.0168 ± 0.0000 | 2.0902 | 0.4784 |  |
| delta_cog__pruned_dataset_dtw_matrix | 0.0136 ± 0.0000 | 0.0152 ± 0.0000 | 1.1120 | 0.8993 |  |
