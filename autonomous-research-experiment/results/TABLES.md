| arm | BLEU | chrF | COMET | % of frozen-oracle headroom |
|---|---|---|---|---|
| A0_greedy | 21.84 | 44.93 | 0.7522 | -30 % |
| A1_beam | 22.85 | 45.89 | 0.7617 | +0 % |
| A3_lenratio | 22.55 | 46.39 | 0.7599 | -9 % |
| A5_mbr_chrf | 23.04 | 46.59 | 0.7609 | +6 % |
| A6_rt_str | 22.29 | 46.02 | 0.7604 | -17 % |
| A7_rt_lat | 22.31 | 45.98 | 0.7614 | -16 % |
| A7c_rt_lat_centred | 22.32 | 46.00 | 0.7613 | -16 % |
| A8_learned_string | 22.09 | 45.88 | 0.7608 | -22 % |
| A8_learned_latent | 22.15 | 45.83 | 0.7607 | -21 % |
| A9_oracle | 26.23 | 48.76 | 0.7923 | +100 % |
| A5_lora_beam8 | 26.50 | 50.08 | 0.7839 | +108 % |
| A5_lora_mbr | 25.81 | 50.39 | 0.7828 | +88 % |
| A5_lora_greedy | 25.66 | 49.06 | 0.7742 | +83 % |

| utility | Spearman vs COMET | top-1 accuracy |
|---|---|---|
| chrF_bt | +0.394 | 0.356 |
| char3_bt | +0.391 | 0.356 |
| lat_cos_mean | +0.250 | 0.333 |
| lat_match_f1 | +0.478 | 0.370 |
| lat_match_pre | +0.465 | 0.366 |
| lat_match_rec | +0.449 | 0.354 |
| lat_cos_mean_c | +0.293 | 0.338 |
| lat_match_f1_c | +0.480 | 0.369 |
| lat_match_pre_c | +0.467 | 0.365 |
| lat_match_rec_c | +0.452 | 0.355 |
| learned_string (MLP, 2433 params) | +0.496 ± 0.001 | 0.353 |
| learned_latent (MLP, 3137 params) | +0.557 ± 0.001 | 0.355 |