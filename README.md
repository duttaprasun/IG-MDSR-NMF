For a low-rank approximation of the data matrix using NMF, we have developed two deep learning models called Input Guided Multiple Deconstruction Single Reconstruction neural network for Non-negative Matrix Factorization (IG-MDSR-NMF) and Input Guided Multiple Deconstruction Single Reconstruction neural network for Relaxed Non-negative Matrix Factorization (IG-MDSR-RNMF), mimicking the the way how humans learn. Both the models are divided into two phases, viz., deconstruction and reconstruction. In the deconstruction phase, the input to the neural network is transformed into the latent space, and in the reconstruction phase, the network tries to regenerate the input from its low-rank representation.

The repository contains five Python files which can be executed using Python commands.

IgMdsrNmf.py

(1) The file contains the implementation of the IG-MDSR-NMF model. (2) The program receives the model_choice, data_choice, and f as input and generates a low-rank representation of the same, along with other supporting files.

IgMdsrRnmf.py

(1) The file contains the implementation of the IG-MDSR-RNMF model. (2) The program receives the model_choice, data_choice, and f as input and generates a low-rank representation of the same, along with other supporting files.

classifying.py

(1) The file contains the implementation of four well-known classification methods, namely K-Nearest Neighbor (KNN), Multilayer Perceptron (MLP), Naive Bayes (NB) and Quadratic Discriminant Analysis (QDA). (2) The file also contains the implementation of four classification performance measures namely, Accuracy (ACC), Cohen-Kappa Score (CKS), F1 Score (FS) and Matthews Correlation Coefficient (MCC). (3) The program receives the model_choice, data_choice, f, and generated low-rank representation of the respective dataset as input and generates a '.xls' file containing all classification performance measures.

clustering.py

(1) The file contains the implementation of four well-known clustering methods, namely Mini Batch k-Means (MBkM), Balanced Iterative Reducing and Clustering utilising Hierarchies (BIRCH), Gaussian Mixture Models (GMM) and Fuzzy c-Means (FcM). (2) The file also contains the implementation of four cluster validity indices namely, Adjusted Mutual Information score (AMI), Adjusted Rand index (ARI), Jaccard index (JI) and Normalized Mutual Information score (NMI). (3) The program receives the model_choice, data_choice, f, and generated low-rank representation of the respective dataset as input and generates a '.xls' file containing all cluster validity scores.

trustworthyness.py

(1) The file contains the implementation of the trustworthyness metric. (2) The program receives the model_choice, data_choice, f, and generated low-rank representation of the respective dataset as input and generates the output value of the metric.

The repository contains five datasets namely, Gastrointestinal Lesions in Regular Colonoscopy (GLRC) dataset, Online News Popularity (ONP) dataset, Parkinson’s Disease Classification (PDC) dataset, Student Performance (SP) dataset, and MovieLens dataset.
