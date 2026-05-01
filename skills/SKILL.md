---
name: kaggle-cli
description: >
  How to use the kaggle CLI — managing datasets, competitions, notebooks,
  models, and benchmarks. Activate this skill when the user asks about
  kaggle CLI commands, workflows, or troubleshooting.
---

# Kaggle CLI

The `kaggle` command-line tool provides access to Kaggle's platform features.

## Installation

```bash
pip install kaggle
```

## Authentication

```bash
# Browser-based OAuth login
kaggle auth login

# Or set KAGGLE_API_TOKEN environment variable
# Or place token in ~/.kaggle/access_token
```

## Command Groups

```
kaggle
├── competitions (alias: c)  — Join and submit to competitions
├── datasets (alias: d)      — Create, download, and manage datasets
├── kernels (alias: k)       — Run and manage notebooks/scripts
├── models (alias: m)        — Upload and version models
└── benchmarks (alias: b)    — Benchmark LLM models on tasks
```

## Specific Tasks

For detailed reference on specific command groups, see:

- [Benchmarks](references/benchmarks.md) — Push tasks, run against LLM models, check status, download results
