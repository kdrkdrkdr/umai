"""UMAI — verbatim engine that inserts fillers [uh]/[um] into the transcription on top of a frozen Qwen3-ASR."""
from .head import VerbatimHead
from .model import UMAI

__version__ = "0.1.0"
__all__ = ["UMAI", "VerbatimHead", "__version__"]
