# JIT-SDP Feature Suite
A feature extraction and analysis framework for Just-In-Time Software Defect Prediction (JIT-SDP).

This repository provides an extensible pipeline for extracting commit-level expert features from software repositories and systematically analyzing their importance and predictive contribution. It extends the canonical 14 change metrics proposed by Kamei et al. with method-level and file-level software metrics.



## Overview

This repository contains two main modules:

- `commit_feature_suite`: feature extraction module
- `jit_sdp_experiment_analysis`: feature analysis and experimental evaluation module

## Repository Structure
commit_feature_suite/
  pyproject.toml
  README.md
  scripts/
    run_shards.py
    validate_rca_json.py
  src/
    commit_feature_suite/
      __main__.py
      cli.py
      config.py
      analyzer.py
      affected/
      features/
      gitops/
      graph/
      metrics/
      output/
      parsers/
      results.py

## Feature Extraction Module

<img src="image/Feature-extraction-workflow.png" width="800">

## Feature Analysis Module

### Supplementary Figures for Section VI. Experimental Results

This section provides the complete versions of Fig. 2 and Fig. 3 reported in Section VI. Experimental Results of the paper.
<img src="image/stage1_consensus_all_features.png" width="600">

<img src="image/all_feature_mdi_npsk_ranks.png" width="600">

