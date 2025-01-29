from slack_sdk.webhook import WebhookClient


slack_webhook = WebhookClient(os.environ["SLACK_WEBHOOK_URL_TESTING"])
