# Prior-work numbers for the comparison table (verified from the source text; 2026-10-11)

Only numbers read in the paper's own text are listed. Sources checked in arXiv versions are marked.

| # | Source | Data | Protocol | Metric | Reported value(s) | Location |
|---|---|---|---|---|---|---|
| 1 | Sarhan et al. 2023 (arXiv 2109.14868v1) | NF-UNSW-NB15-v2 | one attack class held out (zero-day) | Z-DR % | MLP mean 92.45, RF mean 87.37; Fuzzers 76.19 (MLP) / 51.32 (RF) | Tables 4–5, p.11 |
| 1b | same | UNSW-NB15 | same | Z-DR % | MLP mean 85.5, RF 80.67; Fuzzers 20.10 / 14.77 | Tables 2–3 |
| 2 | Wilkie et al. 2026 (arXiv 2601.09902) | LYCOS-IDS2017 | Heartbleed + SQLi held out | AUROC | CLAD open-set .996627; supervised MLP .533859 (Heartbleed .071957) | Table I, p.8 |
| 3 | Nayak 2026 (arXiv 2603.00185) | ToN-IoT | held-out families, 5 seeds | AUC-ROC | zero-day 0.971±0.003; AUC-PR 0.721; Recall@1%FPR 0.783 | Table II, abstract |
| 4 | Zoppi et al. 2023 | 11 datasets | unknown attacks | Rec-Unk | XGBoost 0.38; DEEP classifiers ≤ 0.36 | Table 3, p.8 |
| 5 | Moczkodan & Ragab 2026 (arXiv 2606.11098v2) | CIC-IDS2017 | chronological vs random | macro-F1 | chronological ≈ 0.50 (LSTM/GRU 0.52); random RF 0.77 | §V-A, Table III |
| 6 | Abdessamad & Tazi 2026 | TON-IoT | random vs temporal | macro-F1 / MCC | DT 86.81% → 58.04%; MCC 0.638 → 0.196 | abstract |
| 7 | Cantone et al. 2024 (arXiv 2402.10974v1) | CIC17/18, LycoS17/18 | cross-dataset | AUROC % | CIC17→CIC18: LDA 81.70, DT 59.14, RF 66.93, XGB 80.44; within up to 100.00 | Table 3, p.9 |
| 8 | Talpini et al. 2024 | NF-ToN-IoT-v2 / CICIDS2017 | held-out classes | AUROC (unknown) | NF-ToN-IoT: NN 0.76, Energy 0.80, RF 0.83; CIC: NN 0.68, Energy 0.68, RF 0.69 | Table 8 |
| 9 | Corsini & Yang 2023 (arXiv 2308.14376) | CSE-CIC-IDS2018 | max-softmax at 5% FPR | TPR % | total 53.2 / 33.9 (scenario 1, CL/CE); per-attack as low as 0.0 | Table III |
| 10 | Sarhan et al. 2021 (arXiv 2011.09144) | NF v1 datasets | in-distribution binary | accuracy % | UNSW 98.62, ToN 99.66, BoT 93.82, CSE-CIC 95.33 | Table 8, p.11 |

UNVERIFIED (no number used): Layeghy & Portmann 2023; Alhalangy & Eltayeb 2026; Mesca et al. 2026; Baye et al. 2023;
Zhang et al. 2021; Engelen 2021; Flood 2024. Luay et al. 2025 reports no classification results.
