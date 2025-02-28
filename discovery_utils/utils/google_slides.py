"""Google Slides API utils"""

import os

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from discovery_utils import logging
from discovery_utils.utils.google import Resource
from discovery_utils.utils.google import ServiceAccountCredentials
from discovery_utils.utils.google import find_credentials


EMU_in_CM = 360_000


# Google Drive


def get_drive_service() -> Resource:
    """Initialise Google Drive API service."""
    credentials = get_gslides_credentials()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return service


def upload_image_to_drive(service: Resource, file_path: str) -> tuple[str, str]:
    """Upload a file to Google Drive and makes it public."""

    file_metadata = {"name": os.path.basename(file_path), "mimeType": "image/png"}
    media = MediaFileUpload(file_path, mimetype="image/png")

    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    # Make the file publicly accessible
    service.permissions().create(fileId=uploaded_file["id"], body={"role": "reader", "type": "anyone"}).execute()

    # Get the public URL
    file_id = uploaded_file["id"]
    image_url = f"https://drive.google.com/uc?id={file_id}"

    logging.info(f"Uploaded image available at: {image_url}")

    return file_id, image_url


def get_gslides_credentials():  # noqa
    """Create and return the Google Slides service object."""
    google_credentials_json = find_credentials("GOOGLE_SLIDES_CREDENTIALS")
    scope = ["https://www.googleapis.com/auth/presentations", "https://www.googleapis.com/auth/drive"]
    return ServiceAccountCredentials.from_json_keyfile_name(google_credentials_json, scope)


def get_slides_service() -> Resource:
    """Create and return the Google Slides service object."""
    credentials = get_gslides_credentials()
    service = build("slides", "v1", credentials=credentials, cache_discovery=False)
    return service


def create_slide_from_template(
    template_slide: str,
    new_slide_id: str,
) -> dict:
    """Create a new slide from a template slide"""
    return {
        "duplicateObject": {
            # Existing slide to copy
            "objectId": template_slide,
            # Assign the new slide an ID
            "objectIds": {template_slide: new_slide_id},
        }
    }


def create_image(slide_id: str, image_url: str, image_id: str, transform: dict = None) -> dict:
    """Create image on a slide"""
    transform = transform or {}
    return {
        "createImage": {
            "objectId": image_id,
            "url": image_url,
            "elementProperties": {
                "pageObjectId": slide_id,
                "transform": {
                    "scaleX": transform.get("scaleX", 1),
                    "scaleY": transform.get("scaleY", 1),
                    "translateX": transform.get("translateX", 100000),
                    "translateY": transform.get("translateY", 100000),
                    "unit": "EMU",
                },
            },
        }
    }


def create_textbox(
    text: str,
    slide_id: str,
    text_box_id: str,
    width: int,
    height: int,
    translateX: int,
    translateY: int,
    textbox_formatting: dict = None,
    text_formatting: dict = None,
) -> list[dict]:
    """Create an empty text box on a slide

    Args:
        text (str): The text to insert into the text box.
        text_box_id (str): The unique identifier for the text box.
        slide_id (str): The unique identifier for the slide.
        width (int): The width of the text box.
        height (int): The height of the text box.
        translateX (int): The x-coordinate of the text box.
        translateY (int): The y-coordinate of the text box.
        text_formatting (dict): The text formatting, including following fields:
            - font_family (str): The font family.
            - font_size (int): The font size.
            - bold (bool): Whether the text is bold.
            - italic (bool): Whether the text is italic.
            - underline (bool): Whether the text is underlined.
            - color (list[float]): The RGB color of the text.
    """
    textbox_formatting = {} if textbox_formatting is None else textbox_formatting
    text_formatting = {} if text_formatting is None else text_formatting

    empty_textbox = {
        "createShape": {
            "objectId": text_box_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {"width": {"magnitude": width, "unit": "EMU"}, "height": {"magnitude": height, "unit": "EMU"}},
                "transform": {
                    "scaleX": 1,
                    "scaleY": 1,
                    "translateX": translateX,
                    "translateY": translateY,
                    "unit": "EMU",
                },
            },
        }
    }

    format_textbox = {
        "updateShapeProperties": {
            "objectId": text_box_id,
            "shapeProperties": {"contentAlignment": textbox_formatting.get("content_alignment", "TOP")},
            "fields": "contentAlignment",
        }
    }

    insert_text = {"insertText": {"objectId": text_box_id, "insertionIndex": 0, "text": text}}

    format_text = {
        "updateTextStyle": {
            "objectId": text_box_id,
            "textRange": {"type": "ALL"},
            "style": {
                "fontFamily": text_formatting.get("font_family", "Century Gothic"),
                "fontSize": {"magnitude": text_formatting.get("font_size", 12), "unit": "PT"},
                "bold": text_formatting.get("bold", False),
                "italic": text_formatting.get("italic", False),
                "foregroundColor": {
                    "opaqueColor": {
                        "rgbColor": {
                            "red": text_formatting.get("color", [0.0, 0.0, 0.0])[0],
                            "green": text_formatting.get("color", [0.0, 0.0, 0.0])[1],
                            "blue": text_formatting.get("color", [0.0, 0.0, 0.0])[2],
                        }
                    }
                },
            },
            "fields": "fontFamily,fontSize,bold,italic,foregroundColor",
        }
    }

    format_spacing = {
        "updateParagraphStyle": {
            "objectId": text_box_id,
            "textRange": {"type": "ALL"},
            "style": {"lineSpacing": text_formatting.get("line_spacing", 150)},
            "fields": "lineSpacing",
        }
    }

    return [empty_textbox, format_textbox, insert_text, format_text, format_spacing]


class MissionStudioTemplate:
    """A class to represent a Mission Studio template"""

    def __init__(
        self,
        template_id: str,
        slide_id: str,
        image_url: str,
        heading_text: str,
        details_text: str,
    ) -> None:
        """Initialise the Mission Studio template"""
        self.template_id = template_id
        self.slide_id = slide_id
        self.image_url = image_url
        self.heading_text = heading_text
        self.details_text = details_text

    @staticmethod
    def header_textbox(text: str, slide_id: str) -> dict:
        """Create a header text box"""
        return create_textbox(
            text=text,
            slide_id=slide_id,
            text_box_id=slide_id + "_heading_text_box",
            width=5 * EMU_in_CM,
            height=3 * EMU_in_CM,
            translateX=0.5 * EMU_in_CM,
            translateY=0.5 * EMU_in_CM,
            text_formatting={
                "font_size": 18,
                "bold": True,
                "color": [1.0, 1.0, 1.0],
                "line_spacing": 100,
            },
        )

    @staticmethod
    def details_textbox(text: str, slide_id: str) -> dict:
        """Create a details text box"""
        return create_textbox(
            text=text,
            slide_id=slide_id,
            text_box_id=slide_id + "_details_text_box",
            width=5.5 * EMU_in_CM,
            height=5.5 * EMU_in_CM,
            translateX=0.5 * EMU_in_CM,
            translateY=7 * EMU_in_CM,
            textbox_formatting={
                "content_alignment": "BOTTOM",
            },
            text_formatting={
                "font_size": 11,
                "color": [1.0, 1.0, 1.0],
                "line_spacing": 100,
            },
        )

    @staticmethod
    def major_figure(image_url: str, slide_id: str) -> dict:
        """Create a major figure"""
        return create_image(
            slide_id=slide_id,
            image_url=image_url,
            image_id=slide_id + "_major_figure",
            transform={
                "scaleX": 0.7,
                "scaleY": 0.7,
                "translateX": 7.2 * EMU_in_CM,
                "translateY": 2.0 * EMU_in_CM,
            },
        )

    def slide_request(self) -> None:
        """Create a slide request"""
        return [
            create_slide_from_template(template_slide=self.template_id, new_slide_id=self.slide_id),
            self.header_textbox(self.heading_text, self.slide_id),
            self.details_textbox(self.details_text, self.slide_id),
            self.major_figure(self.image_url, self.slide_id),
        ]


def compare_to_baseline(rate: float, baseline: float, margin: float = 0.1) -> str:
    """Compare a rate to a baseline"""
    if rate > baseline * (1 + margin):
        return "Better"
    elif rate < baseline * (1 - margin):
        return "Worse"
    else:
        return "Similar"


def text_investment(
    chart_type: str,
    investment_types: str,
    category: str,
    n_companies: int,
    n_rounds: int,
    year_start: int,
    year_end: int,
    growth: float,
    growth_start: int,
    growth_end: int,
    baseline_growth: float,
    comparison_margins: float = 0.1,
) -> str:
    """Generate text for a slide with investment round information"""
    comparison = compare_to_baseline(growth, baseline_growth, comparison_margins)

    if chart_type == "funding":
        description = f"Global investment for all companies included in our search results ({investment_types})."
    elif chart_type == "number_of_rounds":
        description = (
            "Number of investment rounds globally for companies included in our search results"
            f" ({investment_types})."
        )
    elif chart_type == "number_of_new_companies":
        description = "Number of new companies founded per year (global)."

    text = (
        f"{description}"
        f"\n\nAll global companies labelled as '{category}' and verified using AI."
        f"\n\n{n_companies} total companies, and {n_rounds} funding rounds in {year_start}-{year_end}."
        f"\n\n{comparison} than baseline: {growth}% growth rate ({growth_start}-{growth_end})"
        f"compared to global baseline {baseline_growth}%"
    )
    return text
