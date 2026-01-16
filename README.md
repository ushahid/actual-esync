# actual-esync
A utility to synchronize transactions with Actual Budget via Emails. Currently it supports Gmail.


# Setup
1. Enable Email notifications in your Bank Account for every transaction. In your Gmail app, filter those messages so that it automatically applies a custom label (label should be unique for every Account)
1. Enable the Gmail API and Obtain the credentails JSON file from Google Clound [[Guide Here](https://developers.google.com/workspace/gmail/api/quickstart/python)]

1. Create a subfolder name private in config
```
mkdir config/private
```

1. Rename the credential file to client_secret.json and move it to private/config
1. Install the requirements
```
pip install -r requirements.txt
```
1. Copy config/config_template.yaml to config/config.yaml
```
cp config/config_template.yaml config/config.yaml
```
1. Edit config/config.yaml to add details of your actual instance (more details in YAML comments, you may skip details under accounts until you want to run sync). WARNING: config.yaml is re-written every time you run sync, any comments will be discarded.
1. If you would like to train the machine learning models to automatically classify payee and categories, run the following command after navigating to the src directory (this might take a few minutes)
```
cd src
python train_eval_clf.py
```

# Synchronization
1. Make sure that you have compelted the steps in Setup and your config file is valid
1. Navigate to the src directory
```
cd src
```
1. Run the following command
```
python sync.py
```
