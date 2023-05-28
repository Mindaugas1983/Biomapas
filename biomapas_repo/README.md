Biomapas
==============================

NLP task for assigning proper answer for given customer request

Project Organization
------------
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for description of this project.
    ├── data
    │   ├── external       <- Data from user questions and answers session - separate user inputs in separate lines
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized FAQ prediction models
    │
    ├── notebooks          <- Jupyter notebooks. None of them are present currently

    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py


--------
Project description
------------
For this task dataset BioASQ11 was used. This dataset is representing medical questions and answers and is possible 
to generate contexts for these questions. However these these question descriptions are not real user context, so in 
this case we can just make a concept how can it be used if real data of user chat sessions was present. Cookiecutter data science template was used for this project in order to keep all files in right order.
Cosine similarity method was used to compare tokenized questions and contexts data.

This project consist of 3 parts and each of them are represented by separate python file:<br /><br />
    1. Extracting FAQ's, answers and contexts from raw data file. Main purpose of this part is to extract only useful data to speed up further development process (this part is represented by make_dataset.py file) <br /><br />
    2. Extracting all valuable information from prepared FAQ data and saving this information and adding function for predicting user inputs (cosine similarity method was used to find most relevant FAQ's) and saving all this information in a file. Main purpose of this part is to separate relatively long model building process from usage of this model.  (this part is represented by train_model.py file) <br /><br />
    3. Loading user chat session inputs from a text file, preparing them, finding most relevant FAQ's for user inputs and printing them in command line. Main purpose of this part is to show how potentially FAQ prediction model can be used (this part is represented by predict_model.py file) <br /><br />

All functions in these python files has docstrings with their functionalty descriptions. For detailed description of this code, you can refer to these docstrings


--------
Usage example
------------

Each python file can be run from command line with such commands:  <br /><br />
make_dataset.py -  python make_dataset.py ..\..\data\raw\nq\training11b.json ..\..\data\processed\medical_data.json     <br /><br />
train_model.py -  python train_model.py ..\..\data\processed\medical_data.json ..\..\models\first   <br /><br />
predict_model.py -  python predict_model.py ..\..\models\first\faq_prediction_model_2023-05-28.pickle ..\..\data\external\input.txt <br /><br />

--------
Prediction results example
------------
User input example: <br /><br />

What is a "Chemobrain" <br />
Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment cancer?<br />
Our analyses indicate that lncRNAs are generated through pathways similar to that of protein-coding genes, with similar histone-modification profiles, splicing signals, and exon/intron lengths. <br />
Describe Multilocus Inherited Neoplasia Allele Syndrome (MINAS) <br /> <br />

Results example: <br /><br />

User input: What is a "Chemobrain"
These FAQ might be relevant for your question:
1. FAQ: What is CPX351?<br />
   Answer: ['CPX-351, a novel liposomal formulation which encapsulates cytarabine and daunorubicin in 5:1 molar ratio, has shown promising efficacy, leading to recent US FDA approval for front-line therapy for patients with therapy-related AML and AML with myelodysplasia-related changes based on a large multicenter Phase III clinical trial.']
2. FAQ: What is a "chemobrain"?<br />
   Answer: ['The term "chemobrain" is sometimes used to denote deficits in neuropsychological functioning that may occur as a result of cancer treatment.']
3. FAQ: What is TFBSshape?<br />
   Answer: ['To utilize DNA shape information when analysing the DNA binding specificities of TFs, the TFBSshape database was developed for calculating DNA structural features from nucleotide sequences provided by motif databases. The TFBSshape database can be used to generate heat maps and quantitative data for DNA structural features (i.e., minor groove width, roll, propeller twist and helix twist) for 739 TF datasets from 23 different species derived from the motif databases JASPAR and UniPROBE. As demonstrated for the basic helix-loop-helix and homeodomain TF families, TFBSshape database can be used to compare, qualitatively and quantitatively, the DNA binding specificities of closely related TFs and, thus, uncover differential DNA binding specificities that are not apparent from nucleotide sequence alone.']<br /><br />
User input: Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment cancer?<br /><br />
These FAQ might be relevant for your question: 
1. FAQ: Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?<br />
   Answer: ['Although is still controversial, Trastuzumab (Herceptin) can be of potential use in the treatment of prostate cancer overexpressing HER2, either alone or in combination with other drugs.']
2. FAQ: Which post-translational histone modifications are characteristic of constitutive heterochromatin?<br />
   Answer: ['H3K9me3 is the major marker of constitutive heterochromatin. Other histone methylation marks usually found in constitutive heterochromatin, are H4K20me3 and H3K79me3. Classical histone modifications associated with heterochromatin include H3K9me2, H3K27me1 and H3K27me2. Histone H3 trimethylation at lysine 36 is associated with constitutive and facultative heterochromatin. H3S10 phosphorylation marks constitutive heterochromatin during interphase in early mouse embryos until the 4-cell stage']
3. FAQ: Which thyroid hormone transporter is implicated in thyroid hormone resistance syndrome?<br />
   Answer: ['thyroid hormone transporter MCT8 is implicated in thyroid hormone resistance syndrome', 'Hemizygous MCT8 mutations cuases TH resistance syndrome in males characterized by severe psychomotor retardation, known as the Allan-Herndon-Dudley syndrome (AHDS).'] <br /><br />
User input: Our analyses indicate that lncRNAs are generated through pathways similar to that of protein-coding genes, with similar histone-modification profiles, splicing signals, and exon/intron lengths. <br /><br />
These FAQ might be relevant for your question: 
1. FAQ: Which methods exist for efficient calculation of Elementary flux modes (EFMs) in genome-scale metabolic networks (GSMNs)?<br />
   Answer: ['EFM-Ta is a novel algorithm that uses a linear programming-based tree search and efficiently enumerates a subset of EFMs in genome-scale metabolic networks (GSMNs). The stand-alone software TreeEFM is implemented in C++ and interacts with the open-source linear solver COIN-OR Linear program Solver (CLP).', 'The efficient calculation of elementary flux modes (EFMs) in genome-scale metabolic networks (GSMNs) is still a challenge. EFM-ta and treeefm are two different algorithms that use linear programming-based tree search and efficiently enumerates a subset of EFMs in GSMNs.', 'Elementary flux modes (EFMs) are a key tool for analyzing genome-scale metabolic networks, and several methods have been proposed to compute them. Among them, those based on solving linear programming (LP) problems like TreeEFM and EFM-Ta are known to be very efficient if the main interest lies in computing large enough sets of EFMs.', 'The efficient calculation of elementary flux modes (EFMs) in genome-scale metabolic networks (GSMNs) is a challenge. Two methods for this task have been developed, eFM-ta and treeefm.']
2. FAQ: Which proteins are related to the loss of cell-cell adhesion during EMT (epithelial-mesenchymal transition)?<br />
   Answer: ['Transcriptional and post-transcriptional regulatory mechanisms mediated by several inducers of EMT, in particular the ZEB and Snail factors, downregulate the expression and/or functional organization of core polarity complexes. Functional loss of the cell-cell adhesion molecule E-cadherin is an essential event for epithelial-mesenchymal transition (EMT), a process that allows cell migration during embryonic development and tumour invasion. Recently, we found that aPKC can also phosphorylate Par6 to drive EMT and increase the migratory potential of non-small cell lung cancer cells. We propose that the regulation of EMT by SIRT1 involves modulation of, and cooperation with, the EMT inducing transcription factor ZEB1. Knockdown of Numb by shRNA in MDCK cells led to a lateral to apical translocation of E-cadherin and beta-catenin, active F-actin polymerization, mis-localization of Par3 and aPKC, a decrease in cell-cell adhesion and an increase in cell migration and proliferation. Growth factors such as TGFb and EGF have also been shown to be related to EMT.']
3. FAQ: Is recommended the use of perioperative treatment with thyroid hormone therapy in patients undergoing coronary artery bypass grafting?<br />
   Answer: ['Currently there is no substantial evidence to justify routine use of thyroid hormones in patients undergoing coronary artery bypass grafting.']<br /><br />
User input: Describe Multilocus Inherited Neoplasia Allele Syndrome (MINAS) <br /><br />
These FAQ might be relevant for your question:
1. FAQ: Describe Multilocus Inherited Neoplasia Allele Syndrome (MINAS)<br />
   Answer: ['Genetic testing of hereditary cancer using comprehensive gene panels can identify patients with more than one pathogenic mutation in high and/or moderate-risk-associated cancer genes. This phenomenon is known as multilocus inherited neoplasia alleles syndrome (MINAS), which has been potentially linked to more severe clinical manifestations.']
2. FAQ: Which growth factors are known to be involved in the induction of EMT?<br />
   Answer: ['EMT is characterized by acquisition of cell motility, modifications of cell morphology, and cell dissociation correlating with the loss of desmosomes from the cellular cortex. A number of growth factors have been shown to be involved in this process. These include fibroblast growth factors (FGFs), TGF-β1, TGF-β2, TNF-α, CCN family, Sonic Hedgehog (SHh), Notch1, GF-β, Wnt, EGF, bFGF, IGF-I and IGF-II.']
3. FAQ: Describe clinical manifestation of the Mal de debarquement syndrome.

   Answer: ['Mal de debarquement syndrome (MdDS) is a disorder of chronic self-motion perception that occurs though entrainment to rhythmic background motion, such as from sea voyage, and involves the perception of low-frequency rocking that can last for months or years.']



--------
Conclusions
------------
This dataset has very complex scientific questions and contexts are not generated from user input. With proper data set model performance might be much better <br />
Model needs to be fine-tuned for particular problem. We need to find proper weights for context and questions similarity. Check what data preprocessing methods are useful and keep only useful ones (most probably text lemmatization, which is used in this project, gives negative impact on accuracy of predictions) <br />
More advanced similarity prediction ways can be implemented (maybe some neural network or other ML algorithm that maps tokenized FAQ and contexts data to question answers) <br />
Making user input checking against misspelling should also benefit in better end user experience <br />
Better user context management algorithm also should provide better results (maybe model answers also should be treated as context, maybe older user inputs potentially has less value than new ones, etc). <br />
