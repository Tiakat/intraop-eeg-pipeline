"""eegpipe - processing pipeline for intraoperative frontal EEG (BIS Vista raw export)."""
from .params import PARAMS, PHASE_ORDER
from .pipeline import process_recording, Result

__all__ = ["PARAMS", "PHASE_ORDER", "process_recording", "Result"]
__version__ = "1.0.0"
