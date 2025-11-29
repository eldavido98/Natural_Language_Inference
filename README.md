# Natural Language Inference

## Overview
This is a simple Python project for experimenting with Natural Language Inference (NLI) — classifying pairs of sentences (premise + hypothesis) into entailment / contradiction / neutral.

The repository contains a minimal implementation of an NLI pipeline: you can feed two sentences (premise and hypothesis) and get an NLI classification. The core logic is in `main.py`.

NOTE: Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), is the task of determining whether a “hypothesis” logically follows (entailment), contradicts, or is neutral given a “premise”.


## Usage
The main script supports two optional modes for each model: **training** and **evaluation**.  
You can enable them via command-line flags.

### Run Training
```python main.py --train_b``` : trains the main model  
```python main.py --train_a``` : trains the secondary model

### Run Evaluation
```python main.py --evaluate_b``` : evaluates the main model  
```python main.py --evaluate_a``` : evaluates the secondary model


## License & Credits  
Feel free to reuse or adapt this code for educational, research or personal purposes.
