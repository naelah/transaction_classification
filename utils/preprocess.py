import spacy
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import fasttext
import pandas as pd
import logging

# from utils.config import config as cf

logging.basicConfig()
logger = logging.getLogger()
logging.getLogger("botocore").setLevel(logging.ERROR)
logger.setLevel(logging.INFO)


class PreProcess:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.col_description = cfg.column_description
        self.col_transaction_description = cfg.column_transaction_description
        self.col_amount = cfg.column_amount

    def preprocess_features(self, df):
        df = self._preprocess_description(df)
        df = self._generate_features(df)
        df = self._generate_category_subcategory_labels(df) # for training only
        return df

    def label_category(self, df):
        df = self._generate_category_labels(df)
        return df

    def label_subcategory(self, df):
        df = self._generate_subcategory_labels(df)
        return df

    def label_category_subcategory(self, df):
        df = self._generate_category_subcategory_labels(df)
        return df

    def remove_min(self, df, type):

        if type == 'category':
            min_samples = self.min_samples_category
            col = self.col_category
        elif type == 'subcategory':
            min_samples = self.min_samples_subcategory
            col = self.col_subcategory
        elif type == 'category_subcategory':
            min_samples = self.min_samples_category_subcategory
            col = self.col_category_subcategory
        print(min_samples)
        freq = df[col].value_counts()
        df = df[~df[col].isin(freq[freq < min_samples].index)].copy().reset_index()

        return df

    def balance_dataset(self, X, y, type):

        if type == 'category':
            n_samples = self.n_samples_category
        elif type == 'subcategory':
            n_samples = self.n_samples_subcategory
        elif type == 'category_subcategory':
            n_samples = self.n_samples_category_subcategory

        # undersample majority classes
        under_sampler = ClusterCentroids(sampling_strategy=sampling_strategy(X, y, n_samples,t='majority'))
        X_under, y_under = under_sampler.fit_resample(X, y)

        # oversample minority classes
        over_sampler = SMOTE(sampling_strategy=sampling_strategy(X_under, y_under,n_samples, t='minority'),k_neighbors=2)
        X_bal, y_bal = over_sampler.fit_resample(X_under, y_under)

        return X_bal, y_bal



        return df

    def _mask_descriptions(self, df):
        return df

    def _remove_descriptions(self, df):
        return df

    def _tokenize_lemmatize(self, column, min_word_len=2):
        """
        Tokenize and Lemmatize text
        :param column:
        :param min_word_len: minimum characters in each word
        :return: column contains list of tokenized words
        """
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
        docs = column.tolist()

        def token_filter(token):
            return not (token.is_punct | token.is_space) and (
                len(token.text) >= min_word_len
            )

        filtered_tokens = []
        for doc in nlp.pipe(docs):
            tokens = [token.lemma_ for token in doc if token_filter(token)]
            filtered_tokens.append(tokens)

        return filtered_tokens

    def _preprocess_description(self, df):
        """
        Preprocess description to remove currency, stop words and tokenize description
        """
        # Combine DESCRIPTION and O_TRANSACTION_DESCRIPTION
        df["T_DESCRIPTION"] = (
            df[self.col_description].fillna("")
            + " "
            + df[self.col_transaction_description].fillna("")
        )
        df["T_DESCRIPTION"] = self._mask_descriptions(df["T_DESCRIPTION"])
        df["T_DESCRIPTION"] = self._remove_descriptions(df["T_DESCRIPTION"])
        df["T_DESCRIPTION"] = self._tokenize_lemmatize(df["T_DESCRIPTION"])
        return df

    def _generate_features(self, df):

        scaler = StandardScaler()
        # AMOUNT is already a string
        df["T_AMOUNT"] = df[self.col_amount].replace(",", "").astype("float")
        df["AMOUNT_LOGABS"] = df["T_AMOUNT"].apply(
            lambda x: np.log(np.abs(x)) if x != 0 else 0
        )
        df["T_AMOUNT"] = scaler.fit_transform(df["AMOUNT_LOGABS"].values.reshape(-1, 1))
        return df

    def _generate_labels(self, df):
        with open(cfg.LABEL_FILE) as f:
            label_map = json.load(f)
        df["CATEGORY_LABEL"] = df.category.map(label_map)
        df["CATEGORY_LABEL"] = df["CATEGORY_LABEL"].astype(float)
        return df

    def _generate_category_subcategory_labels(self, df):
        '''
        generate integer labels for subcategory columns
        '''
        df[self.col_category_subcategory] = df[self.col_category] + '_' + df[self.col_subcategory]
        df[self.col_category_subcategory_label] = self.category_subcategory_to_label(df[self.col_category_subcategory])
        return df
class FastTextVectorizer:
    def __init__(self, cfg):
        self._model = fasttext.FastText.load_model(cfg.FASTTEXT)
        logger.info("Successfully loaded FastText pre trained model")

    def _fit(self, X, y=None):
        return self

    def _transform(self, X):
        result = []
        for words in X:
            sentence = " ".join(words)
            result.append(self._model.get_sentence_vector(sentence))
        return np.array(result)

    def _fit_transform(self, X, y=None):
        self._fit(X, y)
        return self._transform(X)

    def get_embeddings(self, X):
        embeddings = self._fit_transform(X)
        return pd.DataFrame(embeddings)
