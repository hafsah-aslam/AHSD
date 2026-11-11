# Phase 2 — Literature records

**Status (2026-11-11): first pass, 64 papers.** Evidence level per record:
**F** = full text read; **A** = abstract / official page / author summary read (searched this session);
**V** = metadata and claims previously verified against title pages for the HoNE reference check;
**K** = well-known foundational work, details from established knowledge, to be re-read before citing.
Type: **PR** peer-reviewed, **PP** preprint (arXiv / Preprints.org), **PAT** patent (context only).
Fields not yet extracted from full text are marked "—". Full-text reading of the closest competitors to the
candidate directions is in progress (§13) and will upgrade records from A to F. Nothing below is invented:
where a number is given it is the authors' reported number.

## 1. Zero-day / unknown-attack NIDS
| ID | Paper | Type | Problem → method | Data / protocol | Reported result | Limitations (stated / ours) | Ev. |
|---|---|---|---|---|---|---|---|
| Z1 | Wilkie et al., "A novel contrastive loss for zero-day network intrusion detection", IEEE TNSM 23:2064–2076, 2026, doi:10.1109/TNSM.2026.3652529 | PR | zero-day detection with imbalance → contrastive loss (CLAD), open-set extension | Lycos2017; held-out attack classes | AUROC +0.0609 zero-day vs prior; OpenAUC +0.171 | ours: class hold-out only, no temporal novelty (HoNE: CLAD never leads; trails benign-only by 0.075–0.123 on D3) | V |
| Z2 | Nayak, "ThreatFormer-IDS", arXiv:2603.00185, 2026 | PP | robust transformer IDS, zero-day, attribution → weighted supervised + masked SSL + PGD + Integrated Gradients | ToN_IoT; chronological; held-out families | held-out: AUC-PR 0.721, Recall@1%FPR 0.783; chronological AUC 0.994 | single detector; threshold selection not verified | A |
| Z3 | Alhalangy & Eltayeb, "Drift-aware continual generative behavioral detection for zero-day cyberattacks", Information 17(9):852, 2026 (preprint on Preprints.org 202608.0573) | PR | zero-day + drift → DC-GenAIS, leakage-aware temporal protocol, generative replay | — | — | ours: one proposed detector evaluated by its own protocol | A |
| Z4 | Mesca, Pricop, Stamatescu, "Rigorous evaluation of ML intrusion detection for water treatment systems on SWaT network traffic", Appl. Sci. 16(14):7206, 2026 | PR | random splits overstate novelty detection → attack-case and leave-one-attack-out protocols, ensemble with sign-flip of below-chance members | SWaT | — | ours: industrial single testbed | V |
| Z5 | Raj, Tyagi, Kaur, "A leakage-safe re-evaluation of ML intrusion detection for RPL-based IoT networks", J. Eng. Res., in press, 2026, doi:10.1016/j.jer.2026.09.059 | PR | leakage in RPL IDS → future-time and leave-one-attack-type-out tests | RPL simulation | — | — | V |
| Z6 | Zoppi et al., "Which algorithm can detect unknown attacks?…", Comput. Secur. 127:103107, 2023 | PR | which families detect unknowns → benchmark supervised / unsupervised / meta-learning | multiple IDS sets; held-out | meta-learning ensembles best | ours: class hold-out only | V |
| Z7 | Sarhan et al., "From zero-shot ML to zero-day attack detection", Int. J. Inf. Secur. 22(4):947–959, 2023 | PR | why zero-day fails → zero-shot framing; link detection rate to Wasserstein distance | NF datasets; held-out | low zero-day DR correlates with distance to known | ours: HoNE found Wasserstein did not predict detectability (ρ_W −0.18) | V |
| Z8 | Jodayree et al., "Explainable zero-day attack detection in IoMT using transformer-based time-series modeling", Sci. Rep. 16:23252, 2026 | PR | IoMT zero-day → multi-scale transformer with FSM constraints and EVT calibration | IoMT benchmarks | strong open-set (per Sakr et al. summary) | — | V |
| Z9 | Sakr et al., "Explainable and deployment-aware zero-day intrusion detection…EV/V2X", Computers 15(9):599, 2026 | PR | zero-day + leakage → LOAO on 13 families, SHAP/IG audit | CSE-CIC-IDS2018 (= HoNE D1 source); LOAO ×3 | timestamp leakage inflated accuracy up to 20.88 pp; LOAO F1: LR 0.397 > RF 0.239 / ET 0.249 | stated: surrogate data; ours: random LOAO, no temporal novelty | F (text supplied by user) |
| Z10 | Jibunoh, Khanchi, Makanju, "Zero day attacks: novel behaviour or novel vulnerability?", arXiv:2605.03138, 2026 | PP | what zero-days are → incident taxonomy (two decades) | incident records | zero-days are mostly novel vulnerabilities, not novel behaviour | implication: behaviour-based NIDS bounded in what it can detect | A |
| Z11 | Zhao et al., "Towards fine-grained unknown class detection against the open-set attack spectrum with variable legitimate traffic", IEEE/ACM ToN, 2024 | PR | fine-grained unknowns under variable benign → isolation-tree model, incremental updates | — | — | — | A |

## 2. Open-set recognition / unknown-attack identification
| ID | Paper | Type | Problem → method | Data / protocol | Reported result | Limitations | Ev. |
|---|---|---|---|---|---|---|---|
| O1 | Zhang, Zhang, Guo, Song, "A scalable NIDS towards detecting, discovering, and learning unknown attacks", IJMLC 12:1649–1665, 2021 | PR | closed-set IDS misclassify unknowns → OCN (CNN + nearest-class-mean, Fisher + MMD losses), semantic clustering, incremental NCC | KDDCUP'99, CICIDS2017 | outperforms comparison methods | ours: random splits; dated data | A |
| O2 | Baye et al., "Performance analysis of deep-learning based OSR algorithms for NIDS", NOMS 2023, doi:10.1109/NOMS56928.2023.10154410 | PR | benchmark OSR in NIDS → 3 OSR algorithms, hyperparameter sensitivity | OpenSetPerf benchmark | sensitivity to hyperparameters | — | A |
| O3 | Open-set unknown DDoS detection with Reciprocal Points Learning, IEEE Access, 2024 | PR | unknown DDoS → RPL open-set | CICDDoS2019 | avg accuracy up to 98.51% on unknown | ours: accuracy-only, single attack type | A |
| O4 | MI²DAS: multi-layer IDS with incremental learning for IIoT, arXiv:2602.23846, 2026 | PP | OSR + incremental → layered framework | IIoT | macro-F1 0.8995 with new classes | preprint | A |
| O5 | Barbero et al., "Transcending Transcend", IEEE S&P 2022 | PR | drifting malware → classification with rejection via conformal evaluation (ICE, CCE) | 5-year Android malware, time-aware | outperforms SOTA rejection | malware, not network; per-sample rejection | A |

## 3. Out-of-distribution detection (general and in NIDS)
| ID | Paper | Type | Problem → method | Notes | Ev. |
|---|---|---|---|---|---|
| D1 | Hendrycks & Gimpel, ICLR 2017 (MSP baseline) | PR | softmax confidence as OOD score | HoNE: inverts in binary NIDS | K/V |
| D2 | Liu, Wang, Owens, Li, NeurIPS 2020 (energy score) | PR | energy as OOD score; energy-bounded training | HoNE: inverts in binary NIDS | V |
| D3 | Lee et al., NeurIPS 2018 (Mahalanobis) | PR | class-conditional Gaussian on features | HoNE: valid on average, collapses after benign re-anchoring | V |
| D4 | Sun, Ming, Zhu, Li, ICML 2022 (deep kNN) | PR | non-parametric feature-space kNN | as above | V |
| D5 | Yang, Zhou, Li, Liu, "Generalized OOD detection: a survey", IJCV 132(12), 2024 | PR | taxonomy of OOD/OSR/AD | used for HoNE family taxonomy | V |
| D6 | Corsini & Yang, "Are existing OOD techniques suitable for NID?", IEEE CNS 2023 | PR | six OOD techniques on NIDS; binary vs multi-class; center loss | multi-class embeddings better; simple ensembles near-100% in tested scenarios | A |
| D7 | Talpini, Sartori, Savi, "Enhancing trustworthiness in ML-based NID with UQ", J. Reliable Intell. Environ. 10(4), 2024 | PR | uncertainty-aware NIDS vs softmax entropy / energy | uncertainty-aware better | V |
| D8 | Hendrycks, Mazeika, Dietterich, "Deep anomaly detection with outlier exposure", ICLR 2019 | PR | auxiliary outliers improve detection of unseen anomalies | competitor for "benign-anchored" objectives | A |
| D9 | Ruff et al., "Deep semi-supervised anomaly detection" (Deep SAD), ICLR 2020 | PR | labelled anomalies pushed from a hypersphere centre of normal data; entropy view | closest competitor to benign-anchored learning | A |

## 4. Graph neural networks for NIDS
| ID | Paper | Type | Representation / objective | Protocol / result | Limitations | Ev. |
|---|---|---|---|---|---|---|
| G1 | Lo et al., "E-GraphSAGE", IEEE/IFIP NOMS 2022 (arXiv:2103.16329) | PR | endpoints = nodes, flows = edges; GraphSAGE with edge features; supervised edge classification | NF datasets (4 in final version) | supervised, random splits; IP identity as nodes | A |
| G2 | Caville et al., "Anomal-E", Knowledge-Based Systems, 2022 (arXiv:2207.06819) | PR | E-GraphSAGE + modified Deep Graph Infomax, self-supervised | 2 NIDS sets; embeddings > raw features | — | A |
| G3 | Hsieh et al., "NetVigil", USENIX NSDI 2024 | PR | east-west DC flow graphs; GNN + contrastive learning; robust to benign change without attack knowledge | data-centre deployments | DC-specific | A |
| G4 | Guerra, Chapuis, Duc, Mozharovskyi, Nguyen, "Self-supervised learning of graph representations for NID" (GraphIDS), arXiv:2509.16625 v6 (venue claimed NeurIPS 2025, not stated in text read) | PP | hosts (IPs) = nodes, flows = edges; E-GraphSAGE + transformer masked AE over shuffled windows of ≤512 flow embeddings; L = mean‖h−ĥ‖² on benign flows; score s_i = ‖h_i−ĥ_i‖² | NF-UNSW-NB15 v2/v3, NF-CSE-CIC-IDS2018 v2/v3 (20% sample); **random stratified 80/10/10**; threshold/early stopping on labelled validation | up to 0.9998 PR-AUC, 0.9961 macro-F1 (UNSW v3) | random split; no held-out families, no temporal novelty, no fixed-FPR metric; no per-entity state; Infiltration drives misses (summary) | F (partial) |
| G5 | Fu, Li, Xu, "HyperVision", NDSS 2023 | PR | compact flow-interaction graph; unsupervised connectivity / sparsity / statistics | 92 datasets, 48 encrypted attacks; AUC ≥ 0.92, 80.6 Gb/s, 0.83 s | packet-level input | A |
| G6 | Cross-dataset zero-shot IDS aligning graph flow representations with text attack descriptions, AAAI 2025 | PR | graph–text alignment | cross-dataset, unseen types | — | A |
| G7 | Powell, "Role-based lateral movement detection with unsupervised learning", Intell. Syst. Appl. 16:200106, 2022 | PR | infer host roles from connections; flag connections to unusual roles | enterprise data | role-based baseline for peer methods | A |

## 5. Temporal / dynamic graph learning
| ID | Paper | Type | Method | Notes | Ev. |
|---|---|---|---|---|---|
| T1 | King & Huang, "Euler: lateral movement via scalable temporal link prediction", NDSS 2022; ACM TOPS 26(3) 2023 | PR | GNN over snapshots + sequence encoder; anomalous edges by link prediction | host-level, LANL data | A |
| T2 | Liuliakov, Schulz, Hermes, Hammer, "One-class intrusion detection with dynamic graphs" (TGN-SVDD), arXiv:2508.12885, 2025 | PP | IPs = node IDs; TGN per-node memory; min (1/D)Σ‖(z_i⊕z_j)−c‖²+λ‖W‖²; score per event; threshold = 0.99 training percentile | CIC-IDS2017 per day, **temporal** (first 200k events train); F1 0.878/0.964/**0.068**/0.993 (Tue–Fri), ROC-AUC ≥ 0.992; misses Bot and Infiltration; no fixed-FPR metric, no FA guarantee, no sequential test; node-identity shortcut discussed | F |
| T3 | Liuliakov et al., noise-robust one-class ID on dynamic graphs, ESANN 2024 / arXiv:2508.14192 | PR | probabilistic TGN (Gaussian per event) | — | A |
| T4 | Santos Escriche, Nyberg, Kim, Dán, "Channel-centric spatio-temporal graph networks for NID" (CCSTGN), IEEE CNS 2024 | PR | communication channels (host pairs) = nodes (line graph), TGN-style continuous-time node state; supervised, SSL, fine-tuned | NF-UNSW-NB15-v2, temporal order; **thresholds at FPR targets {0.01, 0.1, 1}% set on validation**; a simple MLP is very competitive; no held-out families, no FA guarantee | F |

## 6. Spiking / neuromorphic
| ID | Paper | Type | Notes | Ev. |
|---|---|---|---|---|
| S1 | Wang et al., "Efficient ID model based on convolutional SNN", Sci. Rep. 14:7054, 2024 | PR | CSNN, lightweight IoT, CICIDS2017 | V |
| S2 | Knapp et al., "Efficacy of SNNs for IDS", Cyber-AI 2025, pp. 89–95 | PR | "bare" SNNs vs NNs/RNNs on NSL-KDD, CIC-IDS-2017, CIC-IoT-2023, AWID3 | V |
| S3 | Neuromorphic cybersecurity with semi-supervised lifelong learning, arXiv:2508.04610 (ICONS 2025) | PR(conf.) | static SNN trigger + adaptive SNN classifier; UNSW-NB15; energy by Lava simulation | A |

## 7. Continual / online learning
| ID | Paper | Type | Notes | Ev. |
|---|---|---|---|---|
| C1 | Bas et al., "Adaptive ID for evolving RPL IoT attacks using incremental learning", arXiv:2511.11464 (CCNC 2025) | PR(conf.) | class-incremental; 4× faster than retraining | A |
| C2 | Talpini et al., continual hierarchical BNN NIDS, IEEE NetSoft 2024 | PR | catastrophic forgetting at the edge | A |
| C3 | De Paola, Drago, Ferraro, Lo Re, "HOIDS", J. Netw. Comput. Appl. 254:104556, 2026 | PR | concept-drift-aware hybrid online IDS | V |
| C4 | Amalapuram & Reddy, class-balanced reservoir replay for continual NIDS (IIT Hyderabad) | PR | results depend on task order under imbalance | A |
| C5 | Han et al., "OWAD: anomaly detection in the open world — normality shift detection, explanation, adaptation", NDSS 2023 | PR | unsupervised normality-shift detection and distribution-level adaptation; SCADA deployment | A |

## 8. Self-supervised / contrastive / representation learning of traffic
| ID | Paper | Type | Notes | Ev. |
|---|---|---|---|---|
| R1 | Lin et al., "ET-BERT", WWW 2022 | PR | datagram tokens (BURST, bigrams), pre-trained transformer, encrypted classification | A |
| R2 | netFound, arXiv:2310.17025 (v4 2025) | PP | self-supervised transformer on packet traces; claims temporal generalisation | A |
| R3 | Manocchio et al., "FlowTransformer", ESWA 241:122564, 2024 | PR | transformer framework for flow NIDS | V |
| R4 | Fernández-Morales et al., DL for IoT intrusion detection: SLR, IEEE IoT J 13(20):46485–46513, 2026 | PR | survey | V |
| R5 | "When simple model just works: is network traffic classification in crisis?", arXiv:2506.08655, 2025 | PP | DL adapted from NLP/CV ill-suited; dataset quality | A |
| R6 | Beltiukov et al., "Demystifying network foundation models", NeurIPS 2025 | PR(conf.) | payload-only models miss network statistics; shortcut learning | A |
| R7 | Qing et al., "RAPIER: low-quality training data only?", NDSS 2024 (arXiv:2309.04798) | PR | AE features + label-noise correction; F1 0.77–0.86 at 45% noise | A |

## 9. Sequence modelling / entity- and context-aware detection
| ID | Paper | Type | Notes | Ev. |
|---|---|---|---|---|
| Q1 | Mirsky et al., "Kitsune", NDSS 2018 | PR | AfterImage: damped incremental statistics per channel (per IP, IP-pair, socket) over 5 decay windows; ensemble of AEs | V/A |
| Q2 | Fu et al., "Whisper", ACM CCS 2021 | PR | DFT of per-packet feature sequences, clustering; robust to evasion; AUC 0.999 | A |
| Q3 | Moczkodan & Ragab, "Do transformers actually help ID?", arXiv:2606.11098, 2026 | PP | evaluation choices (padding, leakage-free splits) dominate architecture | A |
| Q4 | Context-aware flow-based NIDS survey, arXiv:2602.05594, 2026 | PP | per-flow models ignore sequential and relational context | A |
| Q5 | BLADE, behaviour-level anomaly detection in web services, arXiv:2511.05193, 2025 | PP | multi-flow behaviour features; CIC-IDS2017 F1 0.973 | A |
| Q6 | Youssef, "Stream assembly is an uncontrolled treatment in streaming ID benchmarks", arXiv:2605.24696 v3, 2026 | PP | audit: stream assembly (order, prevalence, membership) changes results; detector = max(χ² tail, 0.25·BOCPD run-length posterior), stream-level, Elkan cutoff; no conformal martingale, no ARL guarantee, no per-entity accumulation | F (partial) |

## 10. Protocol-aware / dataset quality
| ID | Paper | Type | Notes | Ev. |
|---|---|---|---|---|
| P1 | Engelen, Rimmer, Joosen, CICIDS2017 troubleshooting, IEEE SPW 2021 | PR | flow construction / labelling errors | V |
| P2 | Liu, Engelen, Lynar, Essam, Joosen, "Error prevalence in NIDS datasets: CIC-IDS-2017 and CSE-CIC-IDS-2018", IEEE CNS 2022, pp. 254–263 | PR | undocumented errors incl. labelling; recreated dataset | A |
| P3 | Rosay et al., LYCOS-IDS2017, WI-IAT 2021 | PR | corrected dataset (HoNE D5) | V |
| P4 | Sarhan et al., NetFlow datasets (BDTA 2020) and standard feature set (MONET 2022) | PR | NF-v2 datasets | V |
| P5 | Luay et al., "Time matters", IEEE Access 14:66899–66913, 2026 | PR | NetFlow-v3 temporal features (HoNE data) | V |
| P6 | Pinto et al., ID datasets review, Comput. Netw. 262:111177, 2025 | PR | survey | V |

## 11. Anomaly detection, drift, decision theory
| ID | Paper | Type | Notes | Ev. |
|---|---|---|---|---|
| A1 | Yang et al., "CADE", USENIX Security 2021 | PR | contrastive drift detection; distance-based explanation; NIDS + Android | V |
| A2 | Andresini et al., "INSOMNIA", AISec 2021 | PR | concept-drift robustness in NIDS | V |
| A3 | Pendlebury et al., "TESSERACT", USENIX Security 2019 | PR | temporal and spatial bias in malware evaluation | V |
| A4 | Dragoi et al., "AnoShift", NeurIPS D&B 2022 | PR | distribution-shift benchmark (Kyoto) | V |
| A5 | Schraven et al., "MAWIFlow", ICISSP 2026 | PR | temporally distinct backbone traffic benchmark | V |
| A6 | Siffer et al., "Anomaly detection in streams with EVT" (SPOT/DSPOT), KDD 2017 | PR | GPD peaks-over-threshold thresholds; i.i.d. / drift variant | A |
| A7 | Bates, Candès, Lei, Romano, Sesia, "Testing for outliers with conformal p-values", Ann. Statist. 51(1):149–178, 2023 | PR | marginally valid conformal p-values, PRDS, FDR control, FPR confidence bounds | A |
| A8 | Shin, Ramdas, Rinaldo, "E-detectors: a nonparametric framework for sequential change detection", NEJSDS, 2023/24 (arXiv:2203.03532) | PR | e-CUSUM / e-SR; nonasymptotic ARL control without i.i.d.; near-optimal delay in sub-Gaussian cases | A |
| A9 | Cherubin, Baldwin, Griffin, "Exchangeability martingales for selecting features in anomaly detection", COPA 2018 (PMLR) | PR | conformal martingales; network intrusion keyphrases | A (metadata) |
| A10 | Barrett, Li, Dorai, Rajaganapathy, "FIRCE", arXiv:2605.01962, 2026 | PP | per-flow conformal p-values (class-conditional smoothed ECDF), rejection at quantile τ_c; chunk-level drift from low-confidence outputs; EMA chunk controller; no validity theorem, no ARL, no per-entity accumulation; binary DoS-style attacks; time-to-detection left to future work | F (partial) |
| A11 | Covariance-aware conformal detector for O-RAN intrusion monitoring with false-alarm control, ICML 2026 | PR(conf.) | conformal FA control; per-feature decomposition | A |
| A12 | Tartakovsky et al. / Polunchenko et al., sequential change-point detection for intrusion (multichannel CUSUM, Shiryaev–Roberts), 2006–2012 (arXiv:1212.1829) | PR/PP | foundational quickest detection in NIDS; parametric, volumetric | A |
| A13 | Explainable concept-drift detectors for online IDS: E2D2 (IFIP Networking 2026); Drago et al. (CEUR 2025); Shaer & Shami arXiv:2403.13023 | PR/PP | which features drift; benign reference fingerprints | A |

## 12. Evaluation methodology / XAI / cross-dataset (context)
| ID | Paper | Notes | Ev. |
|---|---|---|---|
| E1 | Arp et al., "Dos and don'ts of ML in computer security", USENIX Security 2022 | pitfalls incl. data snooping, base rates | V |
| E2 | Apruzzese, Laskov, Schneider, "SoK: pragmatic assessment of ML for NID", EuroS&P 2023 | operational assumptions | V |
| E3 | Sommer & Paxson, "Outside the closed world", S&P 2010 | foundational critique | V |
| E4 | Layeghy & Portmann, explainable cross-domain evaluation, Comput. Electr. Eng. 108:108692, 2023 | cross-dataset failure + SHAP | V |
| E5 | Layeghy et al., "DI-NIDS", KBS 273:110626, 2023 | domain-invariant NIDS | V |
| E6 | Cantone et al., cross-dataset generalisation study, IEEE Access 12, 2024 | cross-dataset failure | V |
| E7 | Martínez González et al., cross-domain generalisation for IoT, Internet of Things 39:102044, 2026 | flow-feature cross-domain strategies | V |
| E8 | Liu et al., "NetMasquerade" hard-label black-box evasion, NDSS 2026 | evasion > 96.65% vs 6 detectors | V |
| E9 | Xu et al., "Entente" cross-silo FL on log graphs, NDSS 2026 | federated graph NIDS | V |
| E10 | Grinsztajn et al., tree models on tabular data, NeurIPS 2022 D&B | strong tabular baselines | V |
| E11 | Zhang et al. TESSERACT extended / Kan et al., arXiv 2024 | extended temporal bias analysis | V |
| E12 | HoNE (this group) | protocol validity of zero-day evidence | own |

Patents surfaced (context only, not counted): host-level "concern index" accumulation (US 7475426 / 7185368),
peer-based abnormal host detection (US 10367842), host-based flow aggregation (US 11398987).

## 13. Full-text reading done for the closest competitors
A10 FIRCE, Q6 Youssef, T2 TGN-SVDD, T4 CCSTGN, G4 GraphIDS were read in full (two partially: long tails
summarised). Findings are in their rows and in `../03_gap_matrix_and_opportunities.md` §4. Next pass:
full texts of Z1–Z3, O1, C5 (OWAD), A1 (CADE), A8 (e-detectors), A9, G5 (HyperVision), D9 (Deep SAD).
