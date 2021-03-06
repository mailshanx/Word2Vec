Word2Vec
==============================

Word2vec implementation from scratch using only plain python. Comes with a demo application: 
sentiment analysis on Stanford Sentiment Treebank dataset. 

1. Setup your environment:

   make create_environment

   source activate ws

2. To run sentiment application, type: (Warning: can take a few hours the first time around, depending on your machine).
Make sure you have a data and models folder.

make sentiment


3. To train word vectors, type: (Warning: can take a few hours the first time around, depending on your machine).
Make sure you have a data and models folder.

   make train_word2vec



Other available Make rules:

clean:               Delete all compiled Python files

tests:               1 - button to run tests


See project presentation / slides in docs/w2v_slides.pdf. See original word2vec paper and 
Stanford Sentiment Treebank paper in references.
 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make sentiment` and `make tests`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   │
    │   └── raw            <- The original, immutable data dump. 
    │	  ├─stanfordSetimentTreebank <- Stanford Sentimet Treebank original datadump
    │     ├─get_datasets.sh
    │
    ├── docs               <- Report and presentation
    │ └── _images          <- Images for presentation
    │
    │
    ├── models             <- Trained and serialized models
    │ ├─params             <- Serialized intermediate and final word2vec training params
    │
    │
    ├── references         <- Original word2vec paper and other explanatory materials
    │
    ├── environment.yml    <- Conda environment file: use this to reproduce your python environment
    │
    │
    ├─── w2v               <- Top level python package this project.
       │
       │
       ├── config.py       <- Contains configuration parameters
       │
       │
       ├── word2vec        <- word2vec package: implements word2vec
       │   └── functions.py
       │   └── gradcheck.py
       │   └── sgd.py
       │   └── word2vec.py
       │	
       ├── sentiment       <- Sentiment analysis package: trains word2vec and runs sentiment analysis
       │   └── train_word_vectors.py
       │   └── models.py
       │   └── sentiment.py
       │
       │
       ├── utils           <- Utility scripts: parser for Stanford Sentiment Treebank
       │   └── utils.py                
       │
       └── tests           <- Tests for the project
           └── tests.py
    

--------

