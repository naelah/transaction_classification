# transaction_classification
Predicting categories and subcategories of transactional log based on description

# Dependencies

This repository makes use of pre-trained fasttect word embeddings that can be downloaded here https://fasttext.cc/docs/en/english-vectors.html

# Directory

my-app/
├─ conf/
├─ utils/
│  ├─ wiki-news-300d-1M-subword.bin
│  ├─ dataset_testing.csv
│  ├─ dataset_training.csv
│  ├─ merchants.csv
│  ├─ category_labels.csv
│  ├─ generic_merchants.csv
│  ├─ misc.py
│  ├─ preprocess.py
│  ├─ postprocess.py
│  └── model.py
├─ output/
│  └── models/
│
├─ .gitignore
├─ main.py
├─ requirements.txt.md
└── README.md

