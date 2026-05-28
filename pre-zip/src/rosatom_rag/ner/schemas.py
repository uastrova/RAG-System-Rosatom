from dataclasses import dataclass


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int