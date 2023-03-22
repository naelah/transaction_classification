import os
import pandas as pd
import logging
import hydra
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from utils.preprocess import PreProcess, FastTextVectorizer
from utils.model import ModelServing, ModelTraining
from utils.validate import Expectations
from utils.param_logger import fetch_logged_data
import mlflow
from pprint import pprint


logging.basicConfig()
logger = logging.getLogger()
logging.getLogger("botocore").setLevel(logging.ERROR)
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    all_pass = False
    config_bucket = cfg.data_bucket
    df = pd.read_csv(cfg.dataset.file, delimiter=cfg.dataset.delimiter)
    logger.info(f"Shape of the csv file is: {df.shape}")

    # Validate data, preprocess features, get embeddings
    if not df.empty:
        data = Expectations(df)
        results1, validated1 = data.check_column_exists(cfg.dataset.column_exist)
        results2, validated2 = data.check_unique_values(cfg.dataset.column_unique)
        results3, validated3 = data.check_no_null_values(cfg.dataset.column_no_null)
        all_results = {**results1, **results2, **results3}
        all_pass = all([validated1, validated2, validated3])
    else:
        logger.info("Input dataframe is empty")
        return {"success": False, "message": "Input dataframe is empty"}

    # Labeling category_subcategory
    preprocess = PreProcess(cfg)
    df = preprocess.preprocess_features(df)

    # Continue only if pass validation
    if all_pass:
        X_data = df
        y_data = df[cfg.dataset.column_category_subcategory]

        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_data,
            test_size=cfg.split_test_ratio,
            random_state=cfg.random_state,
        )
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        print(
            "Train :",
            len(X_train),
            " Test :",
            len(X_test),
            " Total :",
            len(X_train) + len(X_test),
        )

 X_train = preprocess.remove_min(X_train, 'category_subcategory').copy().reset_index()
        logger.info("Categories_subcategories with insufficient data is removed")


        # Determine predictor and target columns
        try:
            vectorizer = FastTextVectorizer(cfg)
        except:
            logger.info("Failed to load FastText model")
            return {"success": False, "message": "Failed to load FastText model"}
        else:
            embeddings_train = vectorizer.get_embeddings(X_train["T_DESCRIPTION"]+X_train['T_MCC_DESCRIPTION'])
            X_unbal = embeddings_train
            y_unbal = X_train[cfg.dataset.column_category_subcategory_label]
            logger.info("Embedding category training dataset completed")

        # Balance train data
        X, y = preprocess.balance_dataset(X_unbal, y_unbal, 'category_subcategory')
        logger.info("Balancing category_subcategory training data completed")


        # Training Categories
        trainer = ModelTraining(cfg)
        model = trainer.train_cv(X, y, 'category_subcategory')
        run_id = mlflow.last_active_run().info.run_id
        logger.info("Logged data and model in run {}".format(run_id))

        # Testing Categories
        X_test = preprocess.preprocess_features(X_test)
        logger.info("Preprocessing testing data completed")

        # Label categories
        X_test = preprocess.label_category_subcategory(X_test)
        logger.info("Labeling categories completed")

        embeddings_test = vectorizer.get_embeddings(X_test["T_DESCRIPTION"]+X_test["T_MCC_DESCRIPTION"])
        X = embeddings_test
        y = X_test[cfg.dataset.column_category_subcategory]
        logger.info("Embedding testing dataset completed")

        mlflow.set_experiment(cfg.mlflow_testing_experiment)
        mlflow.sklearn.autolog()

        # Read trained model pickle file from S3
        model = ModelServing(cfg, 'category_subcategory', model=None)
        predicted_df = model.predict_target(X, X_test, 'category_subcategory')

        # Postprocess predicted data
        post_process = PostProcess(
            cfg, predicted_df, params["region"], params["table_name"]
        )
        predicted_processed_df = post_process.enrich_predictions()

         # Exported predicted data
        predicted_processed_df.to_csv(cfg.category_subcategory_output_prediction)
        logger.info("Predicted categories and subcategories successfully exported")



    else:
        logger.info("Data validation has failed")
        return {"success": False, "message": "Data validation has failed"}

if __name__ == "__main__":
    main()