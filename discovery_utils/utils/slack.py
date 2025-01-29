

"""Utilities for building Slack messages with proper block formatting."""

from datetime import date
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import Union

from slack_sdk.webhook import WebhookClient


slack_webhook = WebhookClient(os.environ["SLACK_WEBHOOK_URL_TESTING"])

# Type Definitions
class SlackBlock(TypedDict):
    type: str


class SlackHeaderBlock(SlackBlock):
    type: Literal["header"]
    text: Dict[str, str]


class RichTextStyle(TypedDict, total=False):
    bold: bool
    italic: bool


class RichTextElement(TypedDict, total=False):
    type: Literal["text", "link", "emoji"]
    text: str
    url: str
    style: RichTextStyle
    name: str


class RichTextSection(SlackBlock):
    type: Literal["rich_text_section"]
    elements: List[RichTextElement]


class RichTextListBlock(SlackBlock):
    type: Literal["rich_text_list"]
    style: Literal["bullet"]
    elements: List[RichTextSection]


class RichTextBlock(SlackBlock):
    type: Literal["rich_text"]
    elements: List[Union[RichTextSection, RichTextListBlock]]


class SectionBlock(SlackBlock):
    type: Literal["section"]
    text: Dict[str, str]


# Element Builder
class SlackElement:
    """Builder for Slack message elements."""

    def __init__(self, text: str = ""):
        self.text = text
        self._style: Dict[str, bool] = {}

    def bold(self) -> "SlackElement":
        """Make the element bold."""
        self._style["bold"] = True
        return self

    def italic(self) -> "SlackElement":
        """Make the element italic."""
        self._style["italic"] = True
        return self

    def as_text(self) -> RichTextElement:
        """Build a text element."""
        element: RichTextElement = {"type": "text", "text": self.text}
        if self._style:
            element["style"] = self._style
        return element

    def as_link(self, url: str) -> RichTextElement:
        """Build a link element."""
        element: RichTextElement = {"type": "link", "text": self.text, "url": url}
        if self._style:
            element["style"] = self._style
        return element

    def as_emoji(self) -> RichTextElement:
        """Build an emoji element.

        Note: Country codes should be uppercase (e.g., GB not gb)
        """
        name = self.text
        if name.startswith("flag-"):
            # Ensure country code is uppercase
            prefix, country = name.split("-", 1)
            name = f"{prefix}-{country.upper()}"

        return {"type": "emoji", "name": name}


# Block Builder
class SlackBlock:
    """Builder for Slack message blocks."""

    def create_header(self, text: str, emoji: bool = True) -> SlackHeaderBlock:
        """Create a header block."""
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": emoji,
            },
        }

    def create_section(self, text: str, markdown: bool = True) -> SectionBlock:
        """Create a section block with optional markdown formatting."""
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn" if markdown else "plain_text",
                "text": text,
            },
        }

    def create_rich_text_section(self, elements: List[RichTextElement]) -> RichTextSection:
        """Create a rich text section."""
        return {
            "type": "rich_text_section",
            "elements": elements,
        }

    def create_rich_text_list(self, elements: List[RichTextSection]) -> RichTextListBlock:
        """Create a rich text list block."""
        return {
            "type": "rich_text_list",
            "style": "bullet",
            "elements": elements,
        }

    def create_rich_text_block(self, elements: List[Union[RichTextSection, RichTextListBlock]]) -> RichTextBlock:
        """Create a rich text block."""
        return {
            "type": "rich_text",
            "elements": elements,
        }


class SlackMessage:
    """Builder for complete Slack messages."""

    def __init__(self):
        self.blocks: List[SlackBlock] = []
        self.block_builder = SlackBlock()

    def add_header(self, text: str, emoji: bool = True) -> "SlackMessage":
        """Add a header block to the message."""
        self.blocks.append(self.block_builder.create_header(text, emoji))
        return self

    def add_section(self, text: str, markdown: bool = True) -> "SlackMessage":
        """Add a section block to the message."""
        self.blocks.append(self.block_builder.create_section(text, markdown))
        return self

    def add_rich_text(self, elements: List[Union[RichTextSection, RichTextListBlock]]) -> "SlackMessage":
        """Add a rich text block to the message."""
        self.blocks.append(self.block_builder.create_rich_text_block(elements))
        return self

    def add_divider(self) -> "SlackMessage":
        """Add a divider block to the message."""
        self.blocks.append({"type": "divider"})
        return self

    def build(self) -> List[SlackBlock]:
        """Build the final message blocks."""
        return self.blocks


# Formatting utilities
def format_date(date_obj: date, format_str: str = "%d-%m-%Y") -> str:
    """Format a date object to string."""
    return date_obj.strftime(format_str)


def format_currency(amount: float, currency: str = "£", precision: int = 2) -> str:
    """Format a currency amount."""
    return f"{currency}{amount:,.2f}"
