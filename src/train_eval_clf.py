import os
import argparse
import pandas as pd
import pickle as pkl
import warnings
import yaml
from actual import Actual, queries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import logging
from rich.logging import RichHandler



logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", help="Path to the YAML configuration file", default=os.path.join("..", "config", "config.yaml"))
    parser.add_argument("--model_dir", help="path to the model directory", default=os.path.join("..", "models"))
    args = parser.parse_args()

    logger.info("Loading credentials")
    
    with open(args.config_filepath, "r") as fin:
        cfg = yaml.load(fin, yaml.FullLoader)
    
    with open(cfg["actual"]["actual_pwd_filepath"], "r") as fin:
        actual_pwd = yaml.load(fin, yaml.FullLoader)["pwd"]

    logger.info("Loading transactions")
    transactions = []
    with Actual(base_url=cfg["actual"]["url"], password=actual_pwd, file=cfg["actual"]["file"]) as actual:
        trans = queries.get_transactions(actual.session)
        for tran in trans:
            if tran.category is not None:
                transactions.append({
                    "account": tran.account.name,
                    "description": tran.notes,
                    "amount": tran.amount,
                    "category": tran.category.name,
                    "payee": tran.payee.name,
                    "date": tran.date
                })

    logger.info("Preparing data")

    # Convert to dataframe and split
    df = pd.DataFrame(transactions).dropna()
    df = df.sort_values(by="date", ascending=True)
    split_index = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_index, :], df.iloc[split_index:, :]

    # Prepare categories
    cat_label_enc = LabelEncoder()
    cat_label_enc.fit(df["category"].values)
    

    # Prepare payee
    payee_label_enc = LabelEncoder()
    payee_label_enc.fit(df["payee"].values)
    

    # Encode account
    acc_enc = OneHotEncoder()
    train_acc_feat = acc_enc.fit_transform(train_df["account"].values.reshape(-1, 1)).todense()
    test_acc_feat = acc_enc.transform(test_df["account"].values.reshape(-1, 1)).todense()



    # Encode descriptions
    vec = TfidfVectorizer()
    train_feat = vec.fit_transform(train_df["description"].values).todense()
    X_train = np.asarray(np.hstack([train_feat, train_acc_feat, train_df["amount"].values.reshape(-1, 1)]))
    test_feat = vec.transform(test_df["description"].values).todense()
    X_test = np.asarray(np.hstack([test_feat, test_acc_feat, test_df["amount"].values.reshape(-1, 1)]))
    

    
    # Train category classifier
    y_train = cat_label_enc.transform(train_df["category"].values)
    y_test = cat_label_enc.transform(test_df["category"].values)
    logger.info("Training category classifier")
    cat_clf = RandomForestClassifier(class_weight="balanced")
    cat_clf.fit(X_train, y_train)
    preds = cat_clf.predict(X_test)
    classes = [cat_label_enc.classes_[idx] for idx in np.unique(np.concat([y_test, preds]))]
    logger.info("Evaluating category classifier")
    print(classification_report(y_test, preds, digits=4, target_names=classes))


    # Train payee classifier
    logger.info("Training payee classifier")
    y_train = payee_label_enc.transform(train_df["payee"].values)
    y_test = payee_label_enc.transform(test_df["payee"].values)
    payee_clf = RandomForestClassifier(class_weight="balanced")
    payee_clf.fit(X_train, y_train)
    preds = payee_clf.predict(X_test)
    classes = [payee_label_enc.classes_[idx] for idx in np.unique(np.concat([y_test, preds]))]
    logger.info("Evaluating payee classifier")
    print(classification_report(y_test, preds, digits=4, target_names=classes))


    logger.info("Saving models")
    actual_models = {
        "cat_label_enc": cat_label_enc,
        "payee_label_enc": payee_label_enc,
        "acc_enc": acc_enc,
        "vectorizer": vec,
        "cat_clf": cat_clf,
        "payee_clf": payee_clf
    }

    with open(os.path.join(args.model_dir, "actual_models.pkl"), "wb") as fout:
        pkl.dump(actual_models, fout)
    

    logger.info("Training complete")



if __name__ == "__main__":
    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    main()