ENTITY_TYPES = [
    "ORG",
    "DATE",
    "ADDRESS",
    "NORM_DOC",
    "NORM_REF",
]

LABEL_LIST = ["O"]
for entity_type in ENTITY_TYPES:
    LABEL_LIST.append(f"B-{entity_type}")
    LABEL_LIST.append(f"I-{entity_type}")

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}