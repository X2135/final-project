# Two-stage Faster + Close Datasets

- close criterion: `0.9 <= (drl/two-stage) <= 1.1`
- ratio column `speedup_twostage_over_drl = drl_mean_sec / twostage_mean_sec`

| category | dataset | twostage_mean_sec | drl_mean_sec | speedup_twostage_over_drl | gap_percent |
|---|---|---:|---:|---:|---:|
| twostage_faster | delta_cog__ais_filtered_2021-01-01_dtw_matrix | 0.009028 | 2.335532 | 258.7011 | 25770.11% |
| twostage_faster | delta_cog__iter_00_dtw_matrix | 0.008037 | 0.016800 | 2.0902 | 109.02% |
| twostage_faster | delta_cog__ais_filtered_2021-02-18_dtw_matrix | 0.007267 | 0.013879 | 1.9100 | 91.00% |
| twostage_faster | delta_cog__ais_sample5_dtw_matrix | 0.007488 | 0.013248 | 1.7691 | 76.91% |
| twostage_faster | delta_cog__ais_yuanli_cleaned_dtw_matrix | 0.008625 | 0.014240 | 1.6509 | 65.09% |
| twostage_faster | delta_cog__ais_sample2_pruned_k3_10_dtw_matrix | 0.009794 | 0.014738 | 1.5048 | 50.48% |
| twostage_faster | delta_cog__final_dtw_matrix | 0.010471 | 0.014878 | 1.4208 | 42.08% |
| twostage_faster | delta_cog__ais_filtered_2021-01-11_dtw_matrix | 0.012262 | 0.016690 | 1.3612 | 36.12% |
| twostage_faster | delta_cog__ais_sample2_segments_v2_pruned_k3_10_dtw_matrix | 0.012004 | 0.014873 | 1.2390 | 23.90% |
| twostage_faster | delta_cog__ais_yuanli_dtw_matrix | 0.014833 | 0.018335 | 1.2361 | 23.61% |
| twostage_faster | delta_cog__pruned_dataset_dtw_matrix | 0.013625 | 0.015151 | 1.1120 | 11.20% |
| twostage_faster | delta_cog__ais_sample2_segments_v2_dtw_matrix | 0.013899 | 0.015438 | 1.1108 | 11.08% |
| twostage_faster | delta_cog__ais_filtered_2021-02-24_bbox_lat45_47.5_lon-130_-120_dtw_matrix | 0.016236 | 0.016696 | 1.0283 | 2.83% |
| twostage_faster | delta_cog__clean_ais_filtered_dtw_matrix | 0.015805 | 0.015979 | 1.0110 | 1.10% |
