"""
Transition Matrix Process Drift (TMPD) Library

A comprehensive Python library for detecting and analyzing process drift (concept drift) 
in business processes using transition matrices as a unified data structure.

This library provides:
- Process drift detection using various algorithms (PELT, threshold-based)
- Change point localization and characterization
- Statistical analysis of process changes
- LLM-powered drift understanding and classification
- Support for multiple process perspectives (control-flow, time, resource, data)

Author: Antonio Carlos Meira Neto
License: MIT
"""

from .TMPD_class import TMPD

__version__ = "1.0.0"
__author__ = "Antonio Carlos Meira Neto"
__email__ = "antonio.meira@example.com"

__all__ = ["TMPD"] 