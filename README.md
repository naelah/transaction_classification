# transaction_classification
Predicting categories and subcategories of transactional log based on description

# Dependencies

This repository makes use of pre-trained fasttect word embeddings that can be downloaded here https://fasttext.cc/docs/en/english-vectors.html

# Directory
```
my-app/
├─ conf/
├─ utils/
│  ├─ wiki-news-300d-1M-subword.bin
│  ├─ transaction_dataset_training.csv
│  ├─ transaction_dataset_testing.csv
│  ├─ merchants.csv
│  ├─ category_labels.csv
│  ├─ generic_merchants.csv
│  ├─ misc.py
│  ├─ preprocess.py
│  ├─ postprocess.py
│  └── model.py
├─ output/
│  └── models
│
├─ .gitignore
├─ main.py
├─ requirements.txt.md
└── README.md
```
# Run program

Run `python main.py` after all directories are updated