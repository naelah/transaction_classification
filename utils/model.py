# from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json
from utils.misc import read_joblib, write_joblib
from utils.preprocess import PreProcess
import logging
import joblib

logging.basicConfig()
logger = logging.getLogger()
logging.getLogger("botocore").setLevel(logging.ERROR)
logger.setLevel(logging.INFO)


def keystoint(x):
    return {int(k): v for k, v in x}


class ModelServing:
    def __init__(self, cfg, type, model=None):
        self._cfg = cfg
        if model:
            print("Using current model")
            self._model = model
        else:
            print("Reading model from file")
            if (type=='category'):
                self._model = read_joblib(self._cfg.category_output_model)
            elif (type=='subcategory'):
                self._model = read_joblib(self._cfg.subcategory_output_model)
            elif (type=='category_subcategory'):
                self._model = read_joblib(self._cfg.category_subcategory_output_model)

    def predict_target(self, X, X_test, type):
        y_test_pred = self._model.predict(X)
        y_test_pred_prob = self._model.predict_proba(X)

        if(type == 'category'):

            X_test["CATEGORY_LABEL"] = pd.DataFrame(y_test_pred)
            X_test["CATEGORY_LABEL_STR"] = X_test['CATEGORY_LABEL'].apply(lambda x: str(x))
            preprocess = PreProcess(self._cfg)
            X_test["E_CATEGORY"] = preprocess.label_to_category(X_test['CATEGORY_LABEL_STR'])

            X_test['E_CATEGORY_PROB'] = pd.DataFrame(y_test_pred_prob).max(axis=1)

            X_test["IS_CORRECT_CAT"] = (
                X_test["E_CATEGORY"] == X_test[self._cfg.dataset.column_category]
            )
        elif(type == 'subcategory'):

            X_test["SUBCATEGORY_LABEL"] = pd.DataFrame(y_test_pred)
            X_test["SUBCATEGORY_LABEL_STR"] = X_test['SUBCATEGORY_LABEL'].apply(lambda x: str(x))
            preprocess = PreProcess(self._cfg)
            X_test["E_SUBCATEGORY"] = preprocess.label_to_subcategory(X_test['SUBCATEGORY_LABEL_STR'])

            X_test['E_SUBCATEGORY_PROB'] = pd.DataFrame(y_test_pred_prob).max(axis=1)

            X_test["IS_CORRECT_SUB"] = (
                X_test["E_SUBCATEGORY"] == X_test[self._cfg.dataset.column_subcategory]
            )
        elif(type == 'category_subcategory'):

            X_test["CATEGORY_SUBCATEGORY_LABEL"] = pd.DataFrame(y_test_pred)
            X_test["CATEGORY_SUBCATEGORY_LABEL_STR"] = X_test['CATEGORY_SUBCATEGORY_LABEL'].apply(lambda x: str(x))
            preprocess = PreProcess(self._cfg)
            X_test["E_CATEGORY_SUBCATEGORY"] = preprocess.label_to_category_subcategory(X_test['CATEGORY_SUBCATEGORY_LABEL_STR'])

            X_test['E_CATEGORY_SUBCATEGORY_PROB'] = pd.DataFrame(y_test_pred_prob).max(axis=1)

            X_test["IS_CORRECT_SUB"] = (
                X_test["E_CATEGORY_SUBCATEGORY"] == X_test[self._cfg.dataset.column_subcategory]
            )


        return X_test

    def predict_without_target(self, X, X_test, type):
        y_test_pred = self._model.predict(X)
        y_test_pred_prob = self._model.predict_proba(X)

        if(type == 'category'):

            X_test["CATEGORY_LABEL"] = pd.DataFrame(y_test_pred)
            X_test["CATEGORY_LABEL_STR"] = X_test['CATEGORY_LABEL'].apply(lambda x: str(x))
            preprocess = PreProcess(self._cfg)
            X_test["E_CATEGORY"] = preprocess.label_to_category(X_test['CATEGORY_LABEL_STR'])

            X_test['E_CATEGORY_PROB'] = pd.DataFrame(y_test_pred_prob).max(axis=1)


        elif(type == 'subcategory'):

            X_test["SUBCATEGORY_LABEL"] = pd.DataFrame(y_test_pred)
            X_test["SUBCATEGORY_LABEL_STR"] = X_test['SUBCATEGORY_LABEL'].apply(lambda x: str(x))
            preprocess = PreProcess(self._cfg)
            X_test["E_SUBCATEGORY"] = preprocess.label_to_subcategory(X_test['SUBCATEGORY_LABEL_STR'])

            X_test['E_SUBCATEGORY_PROB'] = pd.DataFrame(y_test_pred_prob).max(axis=1)

        elif(type == 'category_subcategory'):

            X_test["CATEGORY_SUBCATEGORY_LABEL"] = pd.DataFrame(y_test_pred)
            X_test["CATEGORY_SUBCATEGORY_LABEL_STR"] = X_test['CATEGORY_SUBCATEGORY_LABEL'].apply(lambda x: str(x))
            preprocess = PreProcess(self._cfg)
            X_test["E_CATEGORY_SUBCATEGORY"] = preprocess.label_to_subcategory(X_test['CATEGORY_SUBCATEGORY_LABEL_STR'])

            X_test['E_CATEGORY_SUBCATEGORY_PROB'] = pd.DataFrame(y_test_pred_prob).max(axis=1)

        return X_test


class ModelTraining:
    def __init__(self, cfg):
        self._cfg = cfg

    def train_cv(self, X, y, type):
        model = self._train()
        model = self._cv(model, X, y, type)
        return model

    def _train(self):
        if self._cfg.model.name == "svc":
            logger.info(f"Applying model training: {self._cfg.model.name}")
            return SVC(
                C=self._cfg.model.C,
                kernel=self._cfg.model.kernel,
                gamma=self._cfg.model.gamma,
                class_weight=self._cfg.model.class_weight,
                random_state=self._cfg.model.random_state,
                probability=self._cfg.model.probability
            )
        else:
            logger.info("Please spceify training model")

    def _cv(self, model, X, y, type):
        kfold = StratifiedKFold(
            n_splits=self._cfg.kfold,
            shuffle=True,
            random_state=self._cfg.model.random_state,
        )
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
        }
        score = cross_validate(
            model,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        logger.info(
            "Mean 5-Fold Accuracy score: \t{}".format(score["test_accuracy"].mean())
        )
        logger.info(
            "Mean 5-fold Precision score: \t{}".format(+score["test_precision"].mean())
        )
        logger.info(
            "Mean 5-fold Recall score: \t{}".format(+score["test_recall"].mean())
        )
        logger.info("Mean 5-Fold f1 score: \t\t{}".format(score["test_f1"].mean()))
        logger.info(
            "Mean 5-fold Fit time (seconds): {}".format(+score["fit_time"].mean())
        )
        model.fit(X, y)
        if (type == 'category'):
            model_filename = self._cfg.category_output_model
        elif (type == 'subcategory'):
            model_filename = self._cfg.subcategory_output_model
        elif (type == 'category_subcategory'):
            model_filename = self._cfg.category_subcategory_output_model
        write_joblib(model, model_filename)
        logger.info(
            "======= " + self._cfg.model.name + " model exported", model_filename
        )
        return model
