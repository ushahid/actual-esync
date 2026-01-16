import os
import base64
from email.utils import parsedate_to_datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging
from datetime import datetime, UTC


logger = logging.getLogger(__name__)


class GmailService:
    def __init__(self,
                 token_path,
                 creds_path):
        self.token_path = token_path
        self.creds_path = creds_path
        creds = None

        # If modifying these scopes, delete the file token.json
        SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

        # Check for token
        if os.path.exists(self.token_path):
            logger.debug("Token found, loading from file")
            creds = Credentials.from_authorized_user_file(self.token_path,  SCOPES)

        # If token is invalid
        if not creds or not creds.valid:  
            # If refresh possible
            logger.debug("Refreshing token")
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else: # Do the auth flow
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.creds_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

        # Store token
        with open(self.token_path, "w") as token:
            token.write(creds.to_json())
        self.service = build("gmail", "v1", credentials=creds)

        
    def get_label_id(self, label_name):
        results = self.service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        matched_labels = list(filter(lambda x: x["name"] == label_name, labels))
        if len(matched_labels) == 0:
            return None
        return matched_labels[0]["id"]


    def get_messages(self, label_ids=[], query=""):
        RESULTS_PER_PAGE = 500
        results = self.service.users().messages().list(userId="me", labelIds=label_ids, q=query, maxResults=RESULTS_PER_PAGE).execute()
        timestamp = datetime.now(UTC)
        messages = results.get("messages", [])
        next_page_token = results.get("nextPageToken")
        while not next_page_token is None:
            results = self.service.users().messages().list(userId="me", labelIds=label_ids, q=query, maxResults=RESULTS_PER_PAGE, pageToken=next_page_token).execute()
            messages += results.get("messages", [])
            next_page_token = results.get("nextPageToken")
        logger.info(f"Found {len(messages)} messages")

        full_msgs = []
        for message in messages:
            msg = self.service.users().messages().get(userId="me", id=message["id"]).execute()
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            msg_time = parsedate_to_datetime(headers["Date"])
            subject = headers["Subject"]
            text = ""
            html = ""
            body = ""
            if "data" in msg["payload"]["body"]:
                body = base64.urlsafe_b64decode(msg["payload"]["body"]["data"]).decode("utf-8")
            if msg["payload"]["mimeType"] == "multipart/alternative":
                for p in msg["payload"]["parts"]:
                    if p["mimeType"] in ["text/plain"]:
                        data = base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8")
                        text += data
                    elif p["mimeType"]== "text/html":
                        data = base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8")
                        html += data
            # print("===========================================================================================")
            # print(msg_time)
            # print(subject)
            # print(" ======= MSG TEXT =======")
            # print(text)
            # print(" ======= MSG HTML =======")
            # print(html)
            # print(" ======= MSG Body =======")
            # print(body)
            full_msgs.append({
                "time": msg_time,
                "subject": subject,
                "text": text,
                "html": html,
                "body": body
            })
        return timestamp, full_msgs