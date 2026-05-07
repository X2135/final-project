# Final Three-Method Comparison on Delta COG

| Dataset | Method | Model | eps | minPts | K | Noise Ratio | DBI′ | CP′ | SP′ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| k=3_dataset | Adaptive DBSCAN | paper | 0.2087 | 43 | 0 | 1.0000 | nan | nan | nan |
| k=3_dataset | Two-stage | paper | 0.1598 | 2 | 3 | 0.5882 | 1.4324 | 0.1534 | 0.2433 |
| k=3_dataset | DRL | models/drl_delta_plus_batch_mean_s0.pth | 0.008571 | 1 | 48 | 0.0000 | 0.1555 | 0.0060 | 0.0114 |
| k=3(2)_dataset | Adaptive DBSCAN | paper | 0.3259 | 75 | 0 | 1.0000 | nan | nan | nan |
| k=3(2)_dataset | Two-stage | paper | 0.2748 | 4 | 3 | 0.6853 | 1.5061 | 0.2619 | 0.3826 |
| k=3(2)_dataset | DRL | models/drl_delta_plus_batch_mean_s0.pth | 0.018164 | 2 | 6 | 0.6454 | 0.6274 | 0.0591 | 0.0710 |
| k=3(3)_dataset | Adaptive DBSCAN | paper | 0.6898 | 76 | 0 | 1.0000 | nan | nan | nan |
| k=3(3)_dataset | Two-stage | paper | 0.5296 | 6 | 3 | 0.4720 | 1.4152 | 0.4837 | 2.4358 |
| k=3(3)_dataset | DRL | models/drl_delta_plus_batch_mean_s0.pth | 0.021972 | 3 | 4 | 0.4021 | 0.6506 | 0.1381 | 0.2253 |
| k=7_dataset | Adaptive DBSCAN | paper | 0.3460 | 16 | 1 | 0.9053 | nan | nan | nan |
| k=7_dataset | Two-stage | paper | 0.4172 | 3 | 7 | 0.4526 | 1.6765 | 0.4527 | 2.1432 |
| k=7_dataset | DRL | models/drl_delta_plus_batch_mean_s0.pth | 0.035302 | 3 | 6 | 0.4053 | 0.4832 | 0.0681 | 0.1896 |
| k=8_dataset | Adaptive DBSCAN | paper | 0.8472 | 12 | 0 | 1.0000 | nan | nan | nan |
| k=8_dataset | Two-stage | paper | 0.8251 | 2 | 8 | 0.2740 | 1.0501 | 0.6138 | 2.2757 |
| k=8_dataset | DRL | models/drl_delta_plus_batch_mean_s0.pth | 0.067546 | 2 | 7 | 0.2055 | 0.7341 | 0.2144 | 0.2691 |

**Note:** `nan` indicates that the paper-strict metric is undefined because the corresponding Adaptive DBSCAN result collapsed to a degenerate clustering (e.g., all noise).
