# Transition Matrix Process Drift (TMPD)

A comprehensive Python library for process drift detection and analysis using transition matrices as the core data structure.

## Overview

TMPD provides unified process drift detection, change analysis, and process mining utilities with support for multiple perspectives (control-flow, time, resource, and data). The library includes advanced LLM-based characterization capabilities for understanding process changes.

## Key Features

### Process Drift Detection
- **Multi-perspective analysis**: Control-flow, time, resource, and data perspectives
- **Flexible windowing strategies**: Fixed, adaptive, sliding, and continuous windows
- **Statistical change detection**: Multiple detection algorithms and statistical tests
- **Change localization**: Precise identification of what changed in the process

### LLM-Based Characterization
- **Contextualized understanding**: Synthesis and interpretation rather than data repetition
- **Business intelligence**: Transform technical analysis into business insights
- **Flexible configuration**: Customizable information sources via YAML configuration
- **Multi-perspective synthesis**: Connect changes across different process perspectives

### Process Mining Integration
- **BPMN visualization**: Process model comparison and visualization
- **Change pattern recognition**: Identification of known process change patterns

#### Configuration:
The improved characterization uses the `instructions_general_approach.yaml` file with enhanced instructions that focus on:
- **Synthesis over repetition**: Tell a story, don't just list facts
- **Business context**: Explain what changes mean for the business process
- **Connected insights**: Show relationships between different types of changes
- **Selective evidence**: Use data to support insights, not restate them

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from Codes.TMPD_class import TMPD

# Initialize TMPD
tmpd = TMPD(scenario='offline')

# Set up your event log
tmpd.set_transition_log(event_log, case_id='case_id', 
                       activity_key='activity', timestamp_key='timestamp')

# Run transition log processing
tmpd.run_transition_log()

# Configure and run analysis
tmpd.set_windowing_strategy(window_size=100)
tmpd.run_windowing_strategy()

# Run characterization with improved LLM analysis
tmpd.set_characterization_task(llm_company="google", llm_model="gemini-2.0-flash")
tmpd.run_characterization_task()

# Get contextualized results
prompt, response = tmpd.get_characterization_task()
```

## Documentation

### Core Classes
- `TMPD`: Main class for process drift detection and analysis
- `TMPD_process_features`: Process feature extraction utilities
- `TMPD_change_features`: Change detection and analysis utilities
- `TMPD_detection_tasks`: Drift detection algorithms
- `TMPD_understanding_tasks`: LLM-based characterization and understanding

### LLM instruction prompt
- `Codes/LLM_Instructions/instructions_general_approach.yaml`: Main characterization instructions

## Examples

See the `Demonstrations` for comprehensive examples:
- Business Process Drift (Maaradji - Fast)
- CPN Logs Characterization (Ostovar - Robust)
- Real Log Analysis (BPIC 2020)
- Synthetic Event Streams

## Requirements

- Python 3.11.5+
- see requirements.txt

## License

MIT License

## Author

Antonio Carlos Meira Neto