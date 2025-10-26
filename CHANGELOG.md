# Changelog

## [1.0.0] - 2025-10-26

### Added
- **SWE-Bench Verified Integration**: Complete integration with SWE-Bench Verified dataset for software engineering evaluation
- **Dataset Processing**: `create_swebench_verified_tasks.py` script to process and format SWE-Bench Verified tasks
- **Task Definitions**: `inspect_tasks.py` with comprehensive task definitions for different framework categories
- **Sample Dataset**: Pre-processed sample dataset in `inspect_datasets/swebench_verified/combined_samples.json`


### Features
- **Automated Task Creation**
- **Metadata Integration**: Incorporates human baseline completion times from METR evaluation data
- **Flexible Solver Configuration**: Support for both function-calling and non-function-calling models
- **Comprehensive Scoring**: Model-graded evaluation with detailed success criteria


### Usage
```bash
# Generate task dataset
python create_swebench_verified_tasks.py

# Run evaluation on Django tasks
inspect eval inspect_tasks.py@swebench_verified_django --model hf-inference-providers/Qwen/Qwen2.5-Coder-32B-Instruct

# Extract results
unzip logs/yourlog -d results
```

### Technical Details
- **Dataset Source**: Princeton NLP SWE-Bench Verified (HuggingFace)
- **Metadata Source**: METR swe_bench_runs.jsonl
- **Task Format**: Inspect AI Sample format with input/target/metadata structure
- **Evaluation Framework**: Inspect AI with configurable solvers and scorers

### Current Issues & Limitations
- **Model Limitations**: Current Qwen models on HuggingFace only support text generation, not function calling
- **Evaluation Incomplete**: Cannot properly evaluate SWE-Bench Verified tasks without function calling capabilities
- **API Dependencies**: Requires OpenAI and Anthropic API keys for full functionality
- **Solver/Scorer Rework Needed**: Current implementation in inspect_tasks.py needs updates to support real evaluations with function-calling models

### Next Steps
- Obtain OpenAI and Anthropic API keys for proper model access
- Rework solver and scorer configuration in `inspect_tasks.py` for function-calling models
- Implement proper evaluation pipeline for SWE-Bench Verified tasks
- Test with models that support tool use (OpenAI GPT-4, Anthropic Claude, etc.)
