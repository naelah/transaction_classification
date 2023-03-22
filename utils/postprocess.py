import great_expectations as ge
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import sys
from utils.config import cfg as cf

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PostProcess:
    def __init__(self, cfg, df, region, table_name):
        self._df = ge.dataset.PandasDataset(df)
        #self._resource = boto3.resource("dynamodb", region_name=region)
        #self._client = boto3.client("dynamodb", region_name=region)
        #self._table = self._resource.Table(table_name)
        #self._tableName = table_name
        #self._pkey = "merchant_name"
        self._cfg = cfg

        #local files
        self.beu_merchants_dict = cfg.MERCHANTS_FILE
        self.generic_dict = cfg.GENERIC_MERCHANTS_FILE

        #Setting column names

        self._col_t_description = cfg.column_t_description
        self._col_o_merchant = cfg.column_o_merchant
        self._col_subtype = cfg.column_subtype
        self._col_cat_subcat = cfg.column_cat_subcat
        self._col_amount = cfg.column_amount

        self._label_category = cf.category_label
        self._label_subcategory = cf.subcategory_label
        self._category_logo = cf.category_logo
        self._category_income_id = cf.category_income_id
        self._category_income_general = cf.category_income_general
        self._label_remove_general_subcategory = cf.subcategory_remove_general

    def enrich_predictions(self):
        #logger.info(f"Scanning table {self._tableName} to get Merchants")
        #beu_merchants_list = self.scan_beu_merchants()
        #beu_merchants_df = pd.DataFrame(beu_merchants_list)
        #logger.info(f"Merchant table size: {beu_merchants_df.shape}")
        beu_merchants_df = pd.read_csv(self.beu_merchants_dict)
        beu_merchants_df.set_index("merchant_name", inplace=True)
        beu_merchants_df = beu_merchants_df[
            ["regex", "filename", "category", "subcategory", "dominant_color"]
        ]
        beu_merchants_dict = beu_merchants_df.to_dict("index")

        generic_merchants_df = pd.read_csv(self.generic_dict)
        generic_merchants_df.set_index("merchant_name", inplace=True)
        generic_merchants_df = generic_merchants_df[
            ["category", "subcategory"]
        ]
        generic_dict = generic_merchants_df.to_dict("index")
        df = self._df
        #df = self.clean_descriptions(df)
        df = self.set_merchant_name_cat_subcat(beu_merchants_dict, df)
        df = self.mapping_categories_subcategories(df)
        df = self.set_merchant_with_logo(beu_merchants_dict, df)
        df = self.set_generic_cat_subcat(generic_dict, df)
        #df = self.get_logo_colour(df)
        #df = self.clean_merchants(df)
        return df

    def _clean_column(self, column):
        # Apply multiple transformations to the column at once
        column = column.str.strip().str.upper().str.replace(r"[-*@]", " ", regex=True).str.replace(r"\s+", " ", regex=True)
        return column

    def clean_descriptions(self, df):
        df[self._col_t_description] = df["O_TRANSACTION_DESCRIPTION"].fillna(df["DESCRIPTION"]).fillna("Not available")
        df["E_MERCHANT_DESCRIPTION"] = df[self._col_t_description]
        df[self._col_t_description] = self._clean_column(df[self._col_t_description])
        return df

    def set_merchant_name_cat_subcat(self, beu_merchants_dict, df):
        # Create new columns with NaN values
        df[["E_MERCHANT_NAME", "E_MERCHANT_CATEGORY_SUBCATEGORY"]] = np.nan

        # Combine two columns and apply cleaning operations
        df["T_MERCHANT"] = (df[self._col_t_description] + " " + df[self._col_o_merchant].fillna(""))
        df["T_MERCHANT"] = self._clean_column(df["T_MERCHANT"])

        # Apply cleaning operations for specific types of transactions
        #df = self._clean_duitnow_transactions(df, "T_MERCHANT")
        #df = self._clean_transfer_transactions(df, "T_MERCHANT")

        # Loop through merchant dictionary and update relevant columns
        for merchant_name, merch in beu_merchants_dict.items():
            mask = df["T_MERCHANT"].str.contains(merch["regex"], regex=True) & df["E_MERCHANT_NAME"].isna()
            df.loc[mask, "E_MERCHANT_NAME"] = merchant_name
            df.loc[mask, "E_MERCHANT_CATEGORY_SUBCATEGORY"] = merch["category"] + "_" + merch["subcategory"]

        return df

    def set_generic_cat_subcat(self, generic_dict, df):
        # Loop through generic dictionary and update relevant columns
        for merch_name, merch in generic_dict.items():
            mask = df["T_MERCHANT"].str.contains(merch_name, regex=True) & df["E_MERCHANT_NAME"].isna()
            df.loc[mask, "E_MERCHANT_CATEGORY_SUBCATEGORY"] = merch["category"] + "_" + merch["subcategory"]

        return df

    def mapping_categories_subcategories(self, df):
        # SUBTYPE_CAT_MAPPER = "utils/subtype_cat.json"
        # with open(SUBTYPE_CAT_MAPPER) as f:
        #     subtype_cat_mapper = json.load(f)
        subtype_cat_mapper = cf.subtype_cat

        # Mapping categories
        # Post processing INCOME, SAVINGS, filter unidentified Income
        df["T_SUBTYPE"] = df[self._col_subtype].map(subtype_cat_mapper)
        df["E_CATEGORY_SUBCATEGORY"] = df["T_SUBTYPE"].fillna(
            df[self._col_cat_subcat]
        )
        df["E_CATEGORY_LABEL"] = df["E_CATEGORY_SUBCATEGORY"].map(self._label_category)
        unidentified_income_mask = (df[self._col_amount] > 0) & (
            df["E_CATEGORY_LABEL"] != self._category_income_id
        )
        df.loc[
            unidentified_income_mask, "E_CATEGORY_SUBCATEGORY"
        ] = self._category_income_general
        df["E_CATEGORY"] = df["E_CATEGORY_SUBCATEGORY"]
        df["E_CATEGORY_LABEL"] = df["E_CATEGORY_SUBCATEGORY"].map(self._label_category)

        # Mapping subcategories
        df["E_SUBCATEGORY_LABEL"] = df["E_CATEGORY_SUBCATEGORY"].map(
            self._label_subcategory
        )
        df["E_SUBCATEGORY"] = df["E_CATEGORY_SUBCATEGORY"].map(
            self._label_remove_general_subcategory
        )
        return df

    def set_merchant_with_logo(self, beu_merchants_dict, df):
        # Mapping default logo for category first
        df["E_MERCHANT_LOGO"] = df["E_CATEGORY_LABEL"].map(self._category_logo)

        for merchant_name, merch in beu_merchants_dict.items():
            filter_df = df[
                df["T_MERCHANT"].str.contains(merch["regex"], regex=True) == True
            ]
            filter_index = filter_df.index.values.tolist()
            df.loc[filter_index, "E_MERCHANT_LOGO"] = merch["filename"]
        return df

    def scan_beu_merchants(self):
        results = []
        last_evaluated_key = None

        try:
            while True:
                if last_evaluated_key:
                    response = self._table.scan(ExclusiveStartKey=last_evaluated_key)
                else:
                    response = self._table.scan()
                last_evaluated_key = response.get("LastEvaluatedKey")
                results.extend(response["Items"])

                if not last_evaluated_key:
                    break
        except ClientError as e:
            logger.info(e.response["Error"]["Message"])
            results = False
            pass

        return results

    def get_logo_colour(self, df):
        # get colour from logo, temp solution, needs refactor
        default_color = "#ffffff"
        # LOGO_COLOUR_TABLE = "utils/logo_colour.json"
        # with open(LOGO_COLOUR_TABLE) as f:
        #     logo_colour_map = json.load(f)
        logo_colour_map = cfg.logo_colour
        df["T_MERCHANT_LOGO"] = df["E_MERCHANT_LOGO"].str.rsplit("/", n=1).str[-1]
        color_date = re.compile(r"(\d{10,})_")
        df["T_MERCHANT_LOGO"] = df["T_MERCHANT_LOGO"].str.replace(color_date, "")
        df["E_MERCHANT_COLOR_CODE"] = df["T_MERCHANT_LOGO"].map(logo_colour_map)
        df["E_MERCHANT_COLOR_CODE"].fillna(default_color, inplace=True)
        return df

    def _clean_duitnow_transactions(self, df, DESC_COLUMN):
        duit_now = re.compile(r"DuitNow QR to ", flags=re.IGNORECASE)
        duit_now_qr = re.compile(r"DuitNow to ", flags=re.IGNORECASE)
        mask_duitnow = df.TRANSACTION_CHANNEL == "inter_duitnow_out"
        # Case for DuitNow
        df.loc[mask_duitnow, DESC_COLUMN] = df.loc[
            mask_duitnow, DESC_COLUMN
        ].str.replace(duit_now, "", regex=True)
        # Case for DuitNow QR
        df.loc[mask_duitnow, DESC_COLUMN] = df.loc[
            mask_duitnow, DESC_COLUMN
        ].str.replace(duit_now_qr, "", regex=True)
        return df

    def _clean_transfer_transactions(self, df, DESC_COLUMN):
        transfer = re.compile(r"Transfer to ", flags=re.IGNORECASE)
        self_deposit = re.compile(r"Money from ", flags=re.IGNORECASE)
        mask_transfer = df.TRANSACTION_CHANNEL == "intra_beu"
        df.loc[mask_transfer, DESC_COLUMN] = df.loc[
            mask_transfer, DESC_COLUMN
        ].str.replace(transfer, "", regex=True)
        df.loc[mask_transfer, DESC_COLUMN] = df.loc[
            mask_transfer, DESC_COLUMN
        ].str.replace(self_deposit, "", regex=True)
        return df

    def _clean_merchants(self, df):
        df["E_MERCHANT_NAME"] = df["E_MERCHANT_NAME"].str.strip()
        df["E_MERCHANT_NAME"] = df["E_MERCHANT_NAME"].str.title()
        return df

    def clean_merchants(self, df):
        # Populate null values of unidentifiable merchants with transaction description
        df["E_MERCHANT_NAME"].fillna(df["E_MERCHANT_DESCRIPTION"], inplace=True)
        df = self._clean_duitnow_transactions(df, "E_MERCHANT_NAME")
        df = self._clean_transfer_transactions(df, "E_MERCHANT_NAME")
        df = self._clean_merchants(df)
        return df
