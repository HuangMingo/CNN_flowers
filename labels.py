"""Class labels used by the trained flower model.

The order is part of the model contract. Do not sort this collection.
"""

CLASS_NAMES = (
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
)

VIETNAMESE_NAMES = {
    "daisy": "hoa cúc",
    "dandelion": "bồ công anh",
    "rose": "hoa hồng",
    "sunflower": "hướng dương",
    "tulip": "tulip",
}


def display_name(class_name: str) -> str:
    """Return a bilingual label suitable for the user interface."""
    return f"{VIETNAMESE_NAMES[class_name]} ({class_name})"
