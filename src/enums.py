from enum import Enum

GESTURE_NAMES = ["Wave", "Pinch", "Swipe", "Click"]

class ExtractorMode(Enum):
    Magnitude = 'magnitude'
    Realimag = 'realimag'