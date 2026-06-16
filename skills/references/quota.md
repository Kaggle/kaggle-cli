# Quota CLI Reference

Use `kaggle quota` to show the current user's weekly GPU and TPU accelerator
quota for Kaggle kernels.

## Prerequisites

- Python 3.11+ with the `kaggle` package installed.
- Kaggle credentials.

## Command Hierarchy

```text
kaggle quota
```

## `kaggle quota`

Shows current weekly GPU and TPU accelerator quota.

**Usage:**

```bash
kaggle quota [options]
```

**Options:**

- `-v, --csv`: Print CSV instead of table.

**Examples:**

```bash
kaggle quota
kaggle quota -v
```

**Purpose:** Inspect used, remaining, total, and refresh time for GPU/TPU quota.

## Output

Table fields:

- `resource`: `GPU` or `TPU`.
- `used`: Used time in hours.
- `remaining`: Remaining time in hours.
- `total`: Total allowed time in hours.
- `refreshAt`: Quota refresh timestamp when returned by the server.

CSV header:

```text
resource,used,remaining,total,refreshAt
```

## Behavior Details

- Missing GPU or TPU quota entries are skipped.
- If no quota information is returned, it prints `No quota information available`.
- Remaining time is clamped at `0.00h` when used time exceeds total time.

## Notes

- This command reports accelerator quota for kernels; it is not a billing or
  storage quota command.
