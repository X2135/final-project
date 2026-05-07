# Traditional Methods Comparison (Tuned) on Selected Datasets

| Dataset | Method | K | Noise Ratio | DBI' | CP' | SP' | silhouette | best_params |
|---|---|---:|---:|---:|---:|---:|---:|---|
| k=3_dataset | K-Means | 3 | 0.0000 | 1.1801 | 0.8534 | 1.5386 | 0.5317 | n_components=8 |
| k=3_dataset | Agglomerative | 3 | 0.0000 | 0.6181 | 0.8877 | 2.3609 | 0.6259 | linkage=average |
| k=3_dataset | Spectral | 3 | 0.0000 | 0.9728 | 0.9089 | 1.7323 | -0.9986 | sigma=0.440844 |
| k=3(2)_dataset | K-Means | 3 | 0.0000 | 1.4264 | 0.8481 | 1.3192 | 0.3199 | n_components=16 |
| k=3(2)_dataset | Agglomerative | 3 | 0.0000 | 0.6787 | 1.0727 | 1.5854 | 0.2380 | linkage=single |
| k=3(2)_dataset | Spectral | 3 | 0.0000 | 1.4251 | 0.8784 | 1.3835 | -0.8207 | sigma=0.604468 |
| k=3(3)_dataset | K-Means | 3 | 0.0000 | 1.3371 | 1.2154 | 1.4958 | 0.3540 | n_components=5 |
| k=3(3)_dataset | Agglomerative | 3 | 0.0000 | 0.5393 | 1.8288 | 3.2492 | 0.4118 | linkage=average |
| k=3(3)_dataset | Spectral | 3 | 0.0000 | 1.4035 | 1.2502 | 1.6205 | -0.3434 | sigma=2.237613 |
| k=7_dataset | K-Means | 7 | 0.0000 | 1.2315 | 0.7932 | 1.1152 | 0.3436 | n_components=8 |
| k=7_dataset | Agglomerative | 7 | 0.0000 | 0.8367 | 1.1810 | 1.3359 | -0.0553 | linkage=single |
| k=7_dataset | Spectral | 7 | 0.0000 | 1.2509 | 0.8166 | 1.1998 | -0.8105 | sigma=1.772153 |
| k=8_dataset | K-Means | 8 | 0.0000 | 1.2374 | 1.1798 | 1.7202 | 0.3131 | n_components=12 |
| k=8_dataset | Agglomerative | 8 | 0.0000 | 1.0367 | 1.4813 | 1.6304 | 0.1156 | linkage=single |
| k=8_dataset | Spectral | 8 | 0.0000 | 1.2541 | 1.1886 | 1.8340 | -0.6796 | sigma=2.092261 |
