---
categories: Machine Learning
title: Deep EHR Survey
date: 2018-12-18 19:36:26
tags: [EHR, Machine Learning]
---

This post is the reading notes of paper [1], because I'd like to learn some machine learning work on EHR recently.

# EHR Information Extraction(IE)
Each patient encounter is associated with several clinical notes, such as admission notes, discharge summaries, and transfer orders.

### Single concepte extraction
- The most fundamental task involving clinical free text is the extraction of structured medical conceptes, such as diseases, treatments, or procedures.
   - Jagannatha et al.[2,3] treat the concept extraction problem as a sequence labeling task. They experiment with several deep architectures based on RNNs, including LSTMs and GRUs, bidirectional LSTMs(Bi-LSTM), and various combinatons of LSTMs with traditional CRF.
- Other applications of deep learning to clinical concept extraction include named entity recognition(NER).
   - Wu et al.[4] apply pre-trained word embeddings on Chinese clinical text using a CNN.

### Temporal Event Extraction
- This subtask tackles the most complex issue of assigning notions of time to each extracted EHR concept.
   - Fries[5] devised a framework to extract medical events and their corresponding times from clinical notes using a standard RNN.
   - Fries[6] utilizes Stanford's DeepDive application for structured relationships and predictions.

### Relation Extraction
- Relation extraction deals with structured relationships between medical concepts in free text, including relations such as *treatment X improves/worsens/causes condition Y*, or *test X reveals medical probem Y*.
   - Lv et al.[7] use standard text pre-processing methods and UMLS-based word-to-concept mappings in conjuncton with sparse autoencoders to generate features for input to a CRF classifier.

### Abbreviation Expansion
- Each abbreviation can bave tens of possible explanations.
   - Liu et al.[[8]] tackle the roblem by utilizing word embedding approaches.

# EHR Representation Learning
Recent deep learning approaches have been used to project discrete codes into vector space for more detailed analysis and more precise predictive tasks.

### Concept Representation
Several recent studies have applied deep unsupervsed representation learning techniques to derive EHR concept vectors that capture the latent similairities and natural clusters between medical concepts.
- Distributed Embedding: Several researchers have applied NLP techniques for summarizing sparse medical codes into a fixed-size and compressed vector format, such as skip-gram[9,10,13]
- Latent Encoding: Aside from NLP-inspired methods, other common deep learning representation learning techniques have also been used for representing EHR concepts, such as RBM[11] and AE[7].

### Patient Representation
Several different deep learning methods for obtaining vector representations of patients have been proposed in [11,13,14,15,16].

# Outcome Prediction
- Static Outcome Prediction: The prediction of a certain outcome without consider temporal constraints.
   - Nguyen et al.[9] use distributed representations and several ANN and linear models to predict heart failure.
   - Tran et al[11] derive patient vectors with their modified RBM architecture, then train a logistic regression classifier for suicide risk stratification. 
   - DeepPatient[2] generated patient vectors with a 3-layer autoencoder, then used these vectors with logistic regression classifiers to predict a wide variety of ICD9-based disease diagnoses within a prediction window.
   - Jacobson et al.[16] compared deep unsupervised representation of clinical notes for predicting healthcare-associated infections(HAI), utilizing stacked sparse AEs and stacked RBMs along with a word2vec-based embedding approach.
   - Li et al.[17] used a two-layer DBN for identifying osteoporosis.
- Temporal Outcome Prediction: Either to predict the outcome or onset within a certain time interval or to make a prediction based on time series data.
   - Cheng et al.[18] trained a CNN on temporal matrices of medical codes per patient for predicting the onset of both congestive heart failure(CHF) and chronic obstructive pulmonary disease(COPD).
   - Lipton et al.[19] used LSTM networks for predicting on of 128 diagnoses, using target replication at each time step along with auxiliary targets for less-common diagnostic labels as a form of regularization.
   - Choi et al.'s Doctor AI[13] framework was constructed to model how physicians behave by predicting future disease diagnosis along with corresponding timed medication interventions.
   - Pham et al.'s DeepCare[20] framework also derives clinical concept vectors via a skip-gram embedding approach.
   - Nickerson et al.[21] forecast postoperative responses including postoperative urinary retention (POUR) and temporal patterns of postoperative pain using MLP and LSTM networks to suggest more effective postoperative pain management.
   - Nguyen el al.'s Deepr[9] system uses a CNN for predicting unplanned readmission following discharge.
   - Esteban et al.[22] used deep models for predicting the onset of complications relating to kidney transplantation.
   - Che et al.[23] develop a variation of the recurrent GRU cell(GRU-D) which attempts to better handle missing values in clinical time series

# Computational Phenotyping
To revisit and refine broad ilness and diagnosis definitions and boundaries.
- New Phenotype Discovery
- Improving Existing Definitions

# Clinical Data De-Identification


# Reference
1. Shickel B, Tighe P J, Bihorac A, et al. [Deep EHR: a survey of recent advances in deep learning techniques for electronic health record (EHR) analysis[J]](https://arxiv.org/pdf/1706.03446.pdf). IEEE journal of biomedical and health informatics, 2018, 22(5): 1589-1604.
2. A. N. Jagannatha and H. Yu, [Structured prediction models for RNN based sequence labeling in clinical text,](https://arxiv.org/pdf/1608.00612.pdf) in EMNLP, 2016
3. A. Jagannatha and H. Yu, [Bidirectional Recurrent Neural Networks for Medical Event Detection in Electronic Health Records](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5119627/pdf/nihms823967.pdf), arXiv, pp. 473–482, 2016.
4. Y. Wu, M. Jiang, J. Lei, and H. Xu, [Named Entity Recognition in Chinese Clinical Text Using Deep Neural Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4624324/pdf/nihms-708181.pdf), in Studies in Health Technology and Informatics, vol. 216, 2015, pp. 624–628
5. J. A. Fries, [Brundlefly at SemEval-2016 Task 12: Recurrent Neural Networks vs. Joint Inference for Clinical Temporal Information Extraction](http://www.aclweb.org/anthology/S16-1198), in Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), 2016, pp. 1274–1279.
6. C. De Sa, A. Ratner, C. Re, J. Shin, F. Wang, S. Wu, and C. Zhang, [Incremental knowledge base construction using deepdive](http://www.vldb.org/pvldb/vol8/p1310-shin.pdf), The VLDB Journal, pp. 1–25, 2016.
7. X. Lv, Y. Guan, J. Yang, and J. Wu, [Clinical Relation Extraction with Deep Learning](https://pdfs.semanticscholar.org/7fac/52a9b0f96fcee6972cc6ac4687068442aee8.pdf), International Journal of Hybrid Information Technology, vol. 9, no. 7, pp. 237–248, 2016
8. Y. Liu, T. Ge, K. S. Mathews, H. Ji, D. L. Mcguinness, and C. Science, [Exploiting Task-Oriented Resources to Learn Word Embeddings for Clinical Abbreviation Expansion](http://aclweb.org/anthology/W15-3810), in Proceedings of the 2015 Workshop on Biomedical Natural Language Processing (BioNLP 2015), 2015, pp. 92–97.
9. P. Nguyen, T. Tran, N. Wickramasinghe, and S. Venkatesh, [Deepr: A Convolutional Net for Medical Records](https://arxiv.org/pdf/1607.07519.pdf), arXiv, pp. 1–9, 2016.
10. Y. Choi, C. Y.-I. Chiu, and D. Sontag, “Learning Low-Dimensional Representations of Medical Concepts Methods Background,” in AMIA Summit on Clinical Research Informatics, 2016, pp. 41–50.
11. D. Ravi, C. Wong, F. Deligianni, M. Berthelot, J. A. Perez, B. Lo, and G.-Z. Yang, “Deep learning for health informatics,” IEEE Journal of Biomedical and Health Informatics, 2016.
12. E. Choi, M. T. Bahadori, and J. Sun, [Doctor AI: Predicting Clinical Events via Recurrent Neural Networks](http://nematilab.info/bmijc/assets/170607_paper.pdf), arXiv, pp. 1–12, 2015.
13. T. Tran, T. D. Nguyen, D. Phung, and S. Venkatesh, [Learning vector representation of medical objects via EMR-driven nonnegative restricted Boltzmann machines (eNRBM)](https://core.ac.uk/download/pdf/82350634.pdf), Journal of Biomedical Informatics,vol. 54, pp. 96–105, 2015.
14. D. Ravi, C. Wong, F. Deligianni, M. Berthelot, J. A. Perez, B. Lo, and G.-Z. Yang, [Deep learning for health informatics](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7801947), IEEE Journal of Biomedical and Health Informatics, 2016.
15. E. Choi, A. Schuetz, W. F. Stewart, and J. Sun, “Using recurrent neural network models for early detection of heart failure onset.” Journal of the American Medical Informatics Association : JAMIA, vol. 292, no. 3, pp. 344–350, 2016.
16. O. Jacobson and H. Dalianis, “Applying deep learning on electronic health records in Swedish to predict healthcare-associated infections,” ACL 2016, p. 191, 2016.
17. H. Li, X. Li, M. Ramanathan, and A. Zhang, “Identifying informative risk factors and predicting bone disease progression via deep belief networks,” Methods, vol. 69, no. 3, pp. 257–265, 2014.
18. Y. Cheng, F. Wang, P. Zhang, H. Xu, and J. Hu, “Risk Prediction with Electronic Health Records : A Deep Learning Approach,” in SIAM International Conference on Data Mining SDM16, 2015.
19. Z. C. Lipton, D. C. Kale, C. Elkan, and R. Wetzell, [Learning to Diagnose with LSTM Recurrent Neural Networks](https://arxiv.org/pdf/1511.03677.pdf), ICLR, 2016.
20. T. Pham, T. Tran, D. Phung, and S. Venkatesh, [DeepCare: A Deep Dynamic Memory Model for Predictive Medicine](https://arxiv.org/pdf/1602.00357.pdf), arXiv, no. i, pp. 1–27, 2016.
21.  P. Nickerson, P. Tighe, B. Shickel, and P. Rashidi, “Deep neural network architectures for forecasting analgesic response,” in Engineering in Medicine and Biology Society (EMBC), 2016 IEEE 38th Annual International Conference of the. IEEE, 2016, pp. 2966–2969.
22. C. Esteban, O. Staeck, Y. Yang, and V. Tresp, “Predicting Clinical Events by Combining Static and Dynamic Information Using Recurrent Neural Networks,” arXiv, 2016.
23. Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu,[Recurrent neural networks for multivariate time series with missing values](https://arxiv.org/pdf/1606.01865.pdf), arXiv preprint arXiv:1606.01865, 2016.
