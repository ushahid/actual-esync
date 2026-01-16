import logging
import os
import re
import argparse
import pickle as pkl
import decimal
from zoneinfo import ZoneInfo
from copy import deepcopy
import yaml
from rich.logging import RichHandler
from gmail import GmailService
from bs4 import BeautifulSoup
import numpy as np
from actual import Actual, queries


logger = logging.getLogger(__name__)



def parse_value(message, cfg):
    is_soup = False
    source = cfg["source"]
    if source == "soup":
        source = cfg["soup"]["source"]
        pattern = cfg["soup"]["selector"]
        is_soup = True
    else:
        pattern = cfg["regex"]
    
    if is_soup:
        soup = BeautifulSoup(message[source], "html.parser")
        match = soup.select_one(pattern)
        if match:
            return match.get_text().strip()
        else:
            logger.error(f"Unable to parse with selector :: {pattern}")
            logger.error("================== Source ==================")
            logger.error(message[source])
            logger.error("============================================")
    else:
        # Extract value
        regex_keys = cfg["regex_keys"]
        match = re.search(pattern, message[source])
        if match is None:
            logger.error(f"Unable to parse with regex :: {pattern}")
            logger.error("================== Source ==================")
            logger.error(message[source])
            logger.error("============================================")
            return None
        for key in regex_keys:
            if match.groupdict()[key] is not None:
                value = match.group(key).strip()
        return value


def fetch_transactions(cfg, gmail):
    deposit_regex = cfg["deposit_regex"]
    ignore_regex =  cfg["ignore_regex"]
    transactions = []
    label_id = gmail.get_label_id(cfg["gmail_label"])
    if label_id is None:
        logger.error(f"Label {cfg['gmail_label']} not found")
        return None
    last_sync_timestamp = cfg["last_sync"]
    timestamp, messages = gmail.get_messages(label_ids=[label_id], query=f"after:{int(last_sync_timestamp.timestamp())}")
    logger.info(f"Processing {len(messages)} messages")
    for idx, message in enumerate(messages):
        logger.debug(f"Processing message #{idx}")
        time = message["time"]
        subject = message["subject"]
        logger.debug(f"Subject :: {subject}")
        if time <= last_sync_timestamp:
            logger.error(f"Invalid message timestamp {time} <= {last_sync_timestamp}")
        elif re.match(cfg["valid_subject_regex"], subject):
            # Extract amount
            amount = parse_value(message, cfg["amount"])
            if amount is None:
                return None
            amount = float(amount.replace(",", ""))
            amount *= -1
            logger.debug(f"Amount extracted: {amount}")
                
            # Extract description
            desc = parse_value(message, cfg["description"])
            if desc is None:
                return None
            desc = BeautifulSoup(desc, "html.parser").get_text()
            if deposit_regex is not None and re.match(deposit_regex, desc):
                logger.debug("Marked as deposit")
                amount *= -1
            logger.debug(f"Desc. extracted: {desc}")
            logger.info(f"Transaction: {amount} for {desc} at {time}")
            if ignore_regex is not None and re.match(ignore_regex, desc):
                logger.warning(f"Blacklisted transaction ignored :: {desc}")
            else:
                transactions.append({
                    "time": time,
                    "amount": amount,
                    "desc": desc
                })
        else:
            logger.warning(f"Ignored :: {subject}")
    return timestamp, transactions


def sync_transactions(account_name, transactions, actual_cfg, actual_pwd):
    with Actual(base_url=actual_cfg["url"], password=actual_pwd, file=actual_cfg["file"]) as actual:
        account = queries.get_account(actual.session, account_name)
        inserted = []
        for transaction in transactions:
            payee = None
            if "payee" in transaction:
                payee = transaction["payee"]
            category = None
            if "category" in transaction:
                category = transaction["category"]
            t = queries.create_transaction(
                actual.session,
                transaction["time"].date(),
                account,
                notes=transaction["desc"],
                amount=decimal.Decimal(transaction["amount"]),
                payee=payee,
                category=category
            )
            inserted.append(t)
        actual.commit()


def classify_transactions(transactions, account_name, models):
    updated_transactions = []
    for transaction in transactions:
        updated_transaction = deepcopy(transaction)
        acc_feat = models["acc_enc"].transform([[account_name]]).todense()
        desc_feat = models["vectorizer"].transform([transaction["desc"]]).todense()
        feat = np.asarray(np.hstack([desc_feat, acc_feat, [[transaction["amount"]]]]))
        cat_id = models["cat_clf"].predict(feat)[0]
        payee_id = models["payee_clf"].predict(feat)[0]
        category = models["cat_label_enc"].classes_[cat_id]
        payee = models["payee_label_enc"].classes_[payee_id]
        updated_transaction["category"] = category
        updated_transaction["payee"] = payee
        logger.info(f"Prediction :: Description: {transaction['desc']}, Category: {category}, Payee: {payee}")
        updated_transactions.append(updated_transaction)
    return updated_transactions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", help="Path to the yaml configuration file", default=os.path.join("..", "config", "config.yaml"))
    parser.add_argument("--gmail_creds_filepath", help="Path to the gmail JSON client secret file", default=os.path.join("..", "config", "private", "client_secret.json"))
    parser.add_argument("--gmail_token_filepath", help="Path to the gmail JSON token file", default=os.path.join("..", "config", "private", "token.json"))
    
    args = parser.parse_args()

    logger.info("Loading YAML configuration")
    with open(args.config_filepath, 'rb') as fin:
        cfg = yaml.load(fin.read(), yaml.FullLoader)
    with open(cfg["actual"]["actual_pwd_filepath"], 'rb') as fin:
        actual_pwd_cfg = yaml.load(fin.read(), yaml.FullLoader)

    logger.info("Loading gmail service")
    gmail = GmailService(args.gmail_token_filepath, args.gmail_creds_filepath)


    logger.info("Loading classification models")
    if cfg["actual"]["models"] is not None:
        with open(cfg["actual"]["models"], "rb") as fin:
            actual_models = pkl.load(fin)
    else:
        actual_models = None


    logger.info("Starting transaction sync")
    for account_name in cfg["accounts"]:
        logger.info(f"================== Processing: {account_name} ==================")
        
        timestamp, transactions = fetch_transactions(cfg["accounts"][account_name], gmail)
        if transactions is None:
            logger.error(f"Sync failed for {account_name}")
        else:
            if len(transactions) > 0:
                logger.info("Translating to Actual timezone")
                for transaction in transactions:
                    transaction["time"] = transaction["time"].astimezone(ZoneInfo(cfg["actual"]["TZ"]))
                if actual_models is not None:
                    logger.info("Classifying transactions")
                    transactions = classify_transactions(transactions, account_name, actual_models)
                else:
                    logger.info("Skipping classification, no models found")
                logger.info("Synchronizing with Actual")                
                sync_transactions(account_name, transactions, cfg["actual"], actual_pwd_cfg["pwd"])
            
            logger.info("Updating config file")
            cfg["accounts"][account_name]["last_sync"] = timestamp
            with open(args.config_filepath, "w") as fout:
                yaml.dump(cfg, fout)


if __name__ == '__main__':
    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
    main()