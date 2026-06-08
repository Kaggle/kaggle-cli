---
name: kaggle-cli
description: >
  Use the local Kaggle CLI skill for command guidance, workflows, and
  troubleshooting across competitions, datasets, kernels/notebooks, models,
  model variations and versions, inbox file uploads, forums/discussions,
  benchmarks, configuration, OAuth/API-token authentication, and accelerator
  quota. Activate this skill when the user asks about kaggle CLI commands,
  examples, flags, metadata files, download/upload flows, submissions,
  benchmark tasks, or Kaggle CLI behavior.
---

# Kaggle CLI

Use this skill to answer or operate on the `kaggle` command-line tool. Treat
this skill and its references as the available command guide.

## Quick Start

```bash
pip install kaggle
kaggle --help
```

Authentication options:

```bash
kaggle auth login
# or set KAGGLE_API_TOKEN
# or place an access token in ~/.kaggle/access_token
# or use legacy ~/.kaggle/kaggle.json credentials
```

## Command Tree

```text
kaggle
├── competitions | c
│   ├── list, files, download, submit, submissions, leaderboard
│   ├── team-submissions, episodes, replay, logs, pages
│   └── topics {list, show}, topic-messages
├── datasets | d
│   ├── list, files, download, init, create, version
│   ├── metadata, status, delete
│   └── topics {list, show}
├── kernels | k
│   └── list, files, init, push|update, pull|get, output, status, logs, delete
├── models | m
│   ├── list, init, create, get, update, delete
│   ├── topics {list, show}
│   └── variations | instances | v | i
│       ├── get, init, create, files, list, update, delete
│       └── versions | v {list, create, download, files, delete}
├── files {upload}
├── forums | f {list, topics {list, show}}
├── benchmarks | b
│   ├── auth, init
│   ├── tasks | t {push, run, list, status, download, log|logs, models, delete, publish}
│   └── topics {list, show}
├── config {view, set, unset}
├── auth {login, print-access-token, revoke}
└── quota
```

Note: the CLI accepts aliases such as `kernels get` for `kernels pull` and
`kernels update` for `kernels push`. Do not recommend
`models variations versions init`; use `models variations init` for variation
metadata instead.

## Reference Map

Read only the reference needed for the user's task:

- [Competitions](references/competitions.md) - competition discovery, files, downloads, submissions, leaderboards, simulations, pages, topics.
- [Datasets](references/datasets.md) - dataset search, files, downloads, metadata, create/version/status/delete, topics.
- [Kernels](references/kernels.md) - notebook/script discovery, metadata, push/pull, outputs, status, logs, delete.
- [Models](references/models.md) - model records, metadata, create/get/update/delete, model topics.
- [Model Variations](references/model_variations.md) - create and manage framework-specific model variations.
- [Model Variation Versions](references/model_variations_versions.md) - create, list, download, inspect, and delete variation versions.
- [Files](references/files.md) - inbox uploads, resumable uploads, directory compression behavior.
- [Forums](references/forums.md) - global discussion forums, topics, and comments.
- [Benchmarks](references/benchmarks.md) - benchmark auth/init, task push/run/status/download/log/model flows, benchmark topics.
- [Configuration](references/configuration.md) - config file, default path, proxy, default competition.
- [Authentication](references/auth.md) - OAuth login, access token printing, revocation, token/key sources.
- [Quota](references/quota.md) - weekly GPU/TPU accelerator quota.

## Operating Guidance

- Read only the reference needed for the user's task.
- Prefer `kaggle <group> --help` or `kaggle <group> <command> --help` when a flag
  is uncertain in the installed CLI.
- For metadata files, prefer the relevant `init` command to generate a starter
  file before editing it.
- Do not invent commands that are not listed in this skill. If live `--help`
  output differs, report it as a version-specific difference.
