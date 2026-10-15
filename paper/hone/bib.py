import re
R={
'sommer2010':r"""R.~Sommer and V.~Paxson, ``Outside the closed world: On using machine learning for network intrusion detection,'' in \emph{Proc. IEEE Symp. Security and Privacy (S\&P)}, 2010, pp. 305--316.""",
'arp2022':r"""D.~Arp, E.~Quiring, F.~Pendlebury, A.~Warnecke, F.~Pierazzi, C.~Wressnegger, L.~Cavallaro, and K.~Rieck, ``Dos and don'ts of machine learning in computer security,'' in \emph{Proc. 31st USENIX Security Symp.}, 2022, pp. 3971--3988.""",
'engelen2021':r"""G.~Engelen, V.~Rimmer, and W.~Joosen, ``Troubleshooting an intrusion detection dataset: The CICIDS2017 case study,'' in \emph{Proc. IEEE Security and Privacy Workshops (SPW)}, 2021, pp. 7--12.""",
'apruzzese2023sok':r"""G.~Apruzzese, P.~Laskov, and J.~Schneider, ``SoK: Pragmatic assessment of machine learning for network intrusion detection,'' in \emph{Proc. IEEE European Symp. Security and Privacy (EuroS\&P)}, 2023, pp. 592--614.""",
'wilkie2026':r"""J.~Wilkie, H.~Hindy, C.~Michie, C.~Tachtatzis, J.~Irvine, and R.~Atkinson, ``A novel contrastive loss for zero-day network intrusion detection,'' \emph{IEEE Trans. Netw. Service Manag.}, vol.~23, pp. 2064--2076, 2026.""",
'jodayree2026':r"""M.~Jodayree, A.~Kavoosi~Ghafi, S.~Amiri, \emph{et al.}, ``Explainable zero-day attack detection in IoMT using transformer-based time-series modeling,'' \emph{Sci. Rep.}, vol.~16, Art. no. 23252, 2026.""",
'sarhan2023zeroshot':r"""M.~Sarhan, S.~Layeghy, M.~Gallagher, and M.~Portmann, ``From zero-shot machine learning to zero-day attack detection,'' \emph{Int. J. Inf. Secur.}, vol.~22, no.~4, pp. 947--959, 2023.""",
'layeghy2023xdomain':r"""S.~Layeghy and M.~Portmann, ``Explainable cross-domain evaluation of ML-based network intrusion detection systems,'' \emph{Comput. Electr. Eng.}, vol.~108, Art. no. 108692, 2023.""",
'hendrycks2017':r"""D.~Hendrycks and K.~Gimpel, ``A baseline for detecting misclassified and out-of-distribution examples in neural networks,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2017.""",
'liu2020energy':r"""W.~Liu, X.~Wang, J.~Owens, and Y.~Li, ``Energy-based out-of-distribution detection,'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, 2020, pp. 21\,464--21\,475.""",
'nosek2018':r"""B.~A. Nosek, C.~R. Ebersole, A.~C. DeHaven, and D.~T. Mellor, ``The preregistration revolution,'' \emph{Proc. Natl. Acad. Sci. USA}, vol.~115, no.~11, pp. 2600--2606, 2018.""",
'sarhan2021netflow':r"""M.~Sarhan, S.~Layeghy, N.~Moustafa, and M.~Portmann, ``NetFlow datasets for machine learning-based network intrusion detection systems,'' in \emph{Big Data Technologies and Applications (BDTA 2020)}, LNICST, vol.~371. Springer, 2021, pp. 117--135.""",
'sarhan2022standard':r"""M.~Sarhan, S.~Layeghy, and M.~Portmann, ``Towards a standard feature set for network intrusion detection system datasets,'' \emph{Mobile Netw. Appl.}, vol.~27, no.~1, pp. 357--370, 2022.""",
'luay2025':r"""M.~Luay, S.~Layeghy, S.~Hosseininoorbin, M.~Sarhan, N.~Moustafa, and M.~Portmann, ``Temporal analysis of NetFlow datasets for network intrusion detection systems,'' arXiv:2503.04404, 2025.""",
'sharafaldin2018':r"""I.~Sharafaldin, A.~H. Lashkari, and A.~A. Ghorbani, ``Toward generating a new intrusion detection dataset and intrusion traffic characterization,'' in \emph{Proc. 4th Int. Conf. Inf. Syst. Secur. Privacy (ICISSP)}, 2018, pp. 108--116.""",
'moustafa2015':r"""N.~Moustafa and J.~Slay, ``UNSW-NB15: A comprehensive data set for network intrusion detection systems,'' in \emph{Proc. Military Commun. Inf. Syst. Conf. (MilCIS)}, 2015, pp. 1--6.""",
'moustafa2021':r"""N.~Moustafa, ``A new distributed architecture for evaluating AI-based security systems at the edge: Network TON\_IoT datasets,'' \emph{Sustain. Cities Soc.}, vol.~72, Art. no. 102994, 2021.""",
'koroniotis2019':r"""N.~Koroniotis, N.~Moustafa, E.~Sitnikova, and B.~Turnbull, ``Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset,'' \emph{Future Gener. Comput. Syst.}, vol.~100, pp. 779--796, 2019.""",
'rosay2021':r"""A.~Rosay, F.~Carlier, E.~Cheval, and P.~Leroux, ``From CIC-IDS2017 to LYCOS-IDS2017: A corrected dataset for better performance,'' in \emph{Proc. IEEE/WIC/ACM Int. Conf. Web Intell. Intell. Agent Technol. (WI-IAT)}, 2021, pp. 570--575.""",
'yang2024ood':r"""J.~Yang, K.~Zhou, Y.~Li, and Z.~Liu, ``Generalized out-of-distribution detection: A survey,'' \emph{Int. J. Comput. Vis.}, vol.~132, no.~12, pp. 5635--5662, 2024.""",
'pinto2025':r"""D.~Pinto, I.~Amorim, E.~Maia, and I.~Pra\c{c}a, ``A review on intrusion detection datasets: Tools, processes, and features,'' \emph{Comput. Netw.}, vol.~262, Art. no. 111177, 2025.""",
'catillo2023':r"""M.~Catillo, A.~Pecchia, and U.~Villano, ``Which algorithm can detect unknown attacks? Comparison of supervised, unsupervised and meta-learning algorithms for intrusion detection,'' \emph{Comput. Secur.}, vol.~127, Art. no. 103107, 2023.""",
'mirsky2018':r"""Y.~Mirsky, T.~Doitshman, Y.~Elovici, and A.~Shabtai, ``Kitsune: An ensemble of autoencoders for online network intrusion detection,'' in \emph{Proc. Netw. Distrib. Syst. Secur. Symp. (NDSS)}, 2018.""",
'layeghy2023dinids':r"""S.~Layeghy, M.~Baktashmotlagh, and M.~Portmann, ``DI-NIDS: Domain invariant network intrusion detection system,'' \emph{Knowl.-Based Syst.}, vol.~273, Art. no. 110626, 2023.""",
'martinez2026':r"""B.~A. Mart\'{\i}nez Gonz\'{a}lez, A.~D. Bermejo, M.~P\'{e}rez Sevilla, J.~A. Rinc\'{o}n Arango, and D.~Urda, ``Cross-domain generalization strategies for IoT cyberattack detection using flow-based features,'' \emph{Internet Things}, vol.~39, Art. no. 102044, 2026.""",
'fernandez2026':r"""E.~Fern\'{a}ndez-Morales, J.~C. Garc\'{\i}a-Merino, L.~Tobarra, A.~Robles-G\'{o}mez, and R.~Pastor-Vargas, ``Deep learning approaches for intrusion detection in IoT environments: A systematic literature review,'' \emph{IEEE Internet Things J.}, 2026, doi: 10.1109/JIOT.2026.3719346.""",
'manocchio2024':r"""L.~D. Manocchio \emph{et al.}, ``FlowTransformer: A transformer framework for flow-based network intrusion detection systems,'' \emph{Expert Syst. Appl.}, vol.~241, Art. no. 122564, 2024.""",
'wang2024snn':r"""Z.~Wang, F.~A. Ghaleb, A.~Zainal, M.~M. Siraj, and X.~Lu, ``An efficient intrusion detection model based on convolutional spiking neural network,'' \emph{Sci. Rep.}, vol.~14, Art. no. 7054, 2024.""",
'knapp2025':r"""L.~Knapp, S.~Nitzsche, M.~B\"{o}rsig, A.~Vasilache, I.~Baumgart, and J.~Becker, ``Efficacy of spiking neural networks for intrusion detection systems,'' in \emph{Proc. Int. Conf. Cybersecurity and AI-Based Systems (Cyber-AI)}, 2025, doi: 10.1109/Cyber-AI66431.2025.11233776.""",
'liu2026netmasq':r"""Z.~Liu, Y.~Zhao, Z.~Liu, Q.~Li, C.~Fu, G.~Zhou, and K.~Xu, ``A hard-label black-box evasion attack against ML-based malicious traffic detection systems,'' in \emph{Proc. Netw. Distrib. Syst. Secur. Symp. (NDSS)}, 2026.""",
'xu2026entente':r"""J.~Xu, C.~Li, Y.~Zheng, and Z.~Li, ``Entente: Cross-silo intrusion detection on network log graphs with federated learning,'' in \emph{Proc. Netw. Distrib. Syst. Secur. Symp. (NDSS)}, 2026.""",
'depaola2026':r"""A.~De~Paola, S.~Drago, P.~Ferraro, and G.~Lo~Re, ``HOIDS: Concept drift aware hybrid online intrusion detection system,'' \emph{J. Netw. Comput. Appl.}, vol.~254, Art. no. 104556, 2026.""",
'yang2021cade':r"""L.~Yang, W.~Guo, Q.~Hao, A.~Ciptadi, A.~Ahmadzadeh, X.~Xing, and G.~Wang, ``CADE: Detecting and explaining concept drift samples for security applications,'' in \emph{Proc. 30th USENIX Security Symp.}, 2021, pp. 2327--2344.""",
'lopezdeprado2018':r"""M.~L\'{o}pez de Prado, \emph{Advances in Financial Machine Learning}. Hoboken, NJ, USA: Wiley, 2018.""",
'lee2018':r"""K.~Lee, K.~Lee, H.~Lee, and J.~Shin, ``A simple unified framework for detecting out-of-distribution samples and adversarial attacks,'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, 2018, pp. 7167--7177.""",
'sun2022knn':r"""Y.~Sun, Y.~Ming, X.~Zhu, and Y.~Li, ``Out-of-distribution detection with deep nearest neighbors,'' in \emph{Proc. 39th Int. Conf. Mach. Learn. (ICML)}, 2022, pp. 20\,827--20\,840.""",
'liu2008':r"""F.~T. Liu, K.~M. Ting, and Z.-H. Zhou, ``Isolation forest,'' in \emph{Proc. 8th IEEE Int. Conf. Data Mining (ICDM)}, 2008, pp. 413--422.""",
'scholkopf2001':r"""B.~Sch\"{o}lkopf, J.~C. Platt, J.~Shawe-Taylor, A.~J. Smola, and R.~C. Williamson, ``Estimating the support of a high-dimensional distribution,'' \emph{Neural Comput.}, vol.~13, no.~7, pp. 1443--1471, 2001.""",
'chen2016':r"""T.~Chen and C.~Guestrin, ``XGBoost: A scalable tree boosting system,'' in \emph{Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)}, 2016, pp. 785--794.""",
'grinsztajn2022':r"""L.~Grinsztajn, E.~Oyallon, and G.~Varoquaux, ``Why do tree-based models still outperform deep learning on typical tabular data?'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track}, 2022.""",
}
s=open('main.tex').read()
order=[]
for m in re.finditer(r'\\cite\{([^}]*)\}',s):
    for k in m.group(1).split(','):
        k=k.strip()
        if k not in order: order.append(k)
missing=[k for k in order if k not in R]; unused=[k for k in R if k not in order]
print('cited',len(order),'missing',missing,'unused',unused)
out=['\\begin{thebibliography}{%d}'%len(order)]+['\\bibitem{%s}\n%s'%(k,R[k]) for k in order]+['\\end{thebibliography}']
open('refs.tex','w').write('\n\n'.join(out))
