# Hosting a Competition from the CLI

This page documents the host-facing commands added in kaggle-cli for the new
public competition-creation API endpoints (kagglesdk 0.1.31+):

- [`kaggle competitions init`](#kaggle-competitions-init)
- [`kaggle competitions create`](#kaggle-competitions-create)
- [`kaggle competitions pages create`](#kaggle-competitions-pages-create)
- [`kaggle competitions launch`](#kaggle-competitions-launch)
- [`kaggle competitions data update`](#kaggle-competitions-data-update)

All of these commands require an authenticated session
(`kaggle config set username/password` or an API token).

A typical end-to-end host workflow looks like:

```bash
# 1. Scaffold a metadata file.
kaggle competitions init ./my-comp

# 2. Edit ./my-comp/competition-metadata.json (fill in the INSERT_* placeholders).

# 3. Create the (unlaunched) competition.
kaggle competitions create -p ./my-comp
# → Competition created: https://www.kaggle.com/competitions/my-comp-slug

# 4. Author the description and rules pages.
kaggle competitions pages create my-comp-slug --name description -f ./description.md --publish
kaggle competitions pages create my-comp-slug --name rules -f ./rules.md --publish

# 5. Update the competition data (train.csv, test.csv, sample_submission.csv, ...).
kaggle competitions data update my-comp-slug -p ./data -m "Initial release"

# 6. Launch the competition (now, or schedule a future UTC time).
kaggle competitions launch my-comp-slug --at 2027-01-01T00:00:00Z
```

These commands are independent — for example, you can call `pages create`
on a competition that already exists, or use `launch` on a competition created
via the host wizard.

---

## `kaggle competitions init`

Writes a `competition-metadata.json` template into a folder.

**Usage:**

```bash
kaggle competitions init [folder]
```

**Arguments:**

- `folder` (optional): Where to write `competition-metadata.json`. Defaults to
  the current directory.

**Example:**

```bash
kaggle competitions init ./my-comp
```

The generated file:

```json
{
  "title": "INSERT_TITLE_HERE",
  "slug": "INSERT_SLUG_HERE",
  "briefDescription": "INSERT_BRIEF_DESCRIPTION_HERE",
  "privacy": "PUBLIC",
  "disableKernels": false,
  "hackathon": false,
  "cloneCompetitionId": null,
  "cloneExcludeCompetitionData": null,
  "clonePageNames": null,
  "licenseId": null,
  "organizationId": null,
  "numPrizes": null,
  "restrictLinkToEmailList": null,
  "reward": null
}
```

See [Metadata reference](#competition-metadata-reference) below for what each
field means.

---

## `kaggle competitions create`

Creates a new competition from `competition-metadata.json`. The competition is
created in an unlaunched (staged) state — use
[`kaggle competitions launch`](#kaggle-competitions-launch) to publish it.

**Usage:**

```bash
kaggle competitions create [-p folder]
```

**Options:**

- `-p, --path <folder>`: Folder containing `competition-metadata.json`. Defaults
  to the current directory.

**Example:**

```bash
kaggle competitions create -p ./my-comp
# → Competition created: https://www.kaggle.com/competitions/my-comp-slug
```

**Errors you might see:**

- `Default title detected, please update competition-metadata.json before creating`
  — you forgot to replace one of the `INSERT_*_HERE` placeholders.
- `Invalid privacy '...'` — `privacy` must be one of `PUBLIC`, `LIMITED`, `PRIVATE`.
- `Metadata file not found: competition-metadata.json` — run `init` first, or pass
  `-p` pointing at the folder that contains the file.

### Competition metadata reference

All fields go in `competition-metadata.json` (camelCase keys).

**Required:**

| Field | Type | Notes |
|---|---|---|
| `title` | string | Display title shown on the competition page. |
| `slug` | string | URL slug; lowercase, hyphens, must be unique site-wide and must not be all digits or all hyphens. |
| `briefDescription` | string | One-line subtitle under the title. |
| `privacy` | string | One of `PUBLIC`, `LIMITED`, `PRIVATE`. |

**Optional:**

| Field | Type | Notes |
|---|---|---|
| `disableKernels` | bool | If `true`, notebook submissions are disabled. |
| `hackathon` | bool | Create as a hackathon competition. |
| `restrictLinkToEmailList` | bool | Restrict invite-link joiners to a host-maintained allowlist. |
| `cloneCompetitionId` | int | If set, clone configuration / pages / data / evaluation setup from this competition. |
| `cloneExcludeCompetitionData` | bool | If cloning, skip copying the data (solution, sandbox submissions, images, databundles). |
| `clonePageNames` | string[] | If cloning, copy only these page names. Omit/null to copy all. |
| `licenseId` | int | License ID for the competition data. |
| `organizationId` | int | Tie this competition to an organization (read-only access for all org members). |
| `numPrizes` | int | Number of leaderboard prize positions. |
| `reward` | object | See below. |

**`reward` object:**

```json
{
  "id": "USD",
  "quantity": 25000,
  "clarification": "Total prize pool split across the top 5 teams."
}
```

`reward.id` is one of: `USD`, `KUDOS`, `AUD`, `EUR`, `JOBS`, `SWAG`, `GBP`,
`KNOWLEDGE`, `PRIZES`. `clarification` is optional free-form text shown next to
the prize.

---

## `kaggle competitions pages create`

Creates a new page (description, rules, evaluation, data-description, etc.) on a
competition you host.

**Usage:**

```bash
kaggle competitions pages create <competition> --page-name <page-name> -f <path> \
    [--mime-type <type>] [--post-title "<title>"] [--publish]
```

**Arguments:**

- `<competition>`: The competition slug.

**Options:**

- `--page-name <page-name>` (required): Page name (e.g. `description`, `rules`,
  `evaluation`, `data-description`, `prizes`). Conventional names are
  recognized by the competition page UI; new names are allowed but won't be
  shown in the standard tabs.
- `-f, --file <path>` (required): Path to a file whose contents become the page
  body.
- `--mime-type <type>` (optional): MIME type of the content. Defaults to
  `text/html` server-side.
- `--post-title "<title>"` (optional): Title shown above the page body.
  Defaults to the page name.
- `--publish` (optional): Publish the page immediately. Without this flag the
  page is created in a staged (unpublished) state so you can review it before
  going live.

**Example:**

```bash
# Create the rules page in a staged (not-yet-published) state.
kaggle competitions pages create my-comp --page-name rules -f ./rules.md \
    --mime-type text/markdown --post-title "Competition Rules"

# When you're ready to make it visible to participants:
kaggle competitions pages update my-comp --page-name rules --publish
```

Each page exists as a single record; `--publish` / `--unpublish` toggles its
visibility rather than creating separate draft and live copies. To swap in new
content later, use
[`kaggle competitions pages update`](#kaggle-competitions-pages-update) — a
second `create` for the same page name will be rejected.

You can list and inspect existing pages with `kaggle competitions pages`
(or the explicit `kaggle competitions pages list`), modify one in place with
[`kaggle competitions pages update`](#kaggle-competitions-pages-update), or
remove one with [`kaggle competitions pages delete`](#kaggle-competitions-pages-delete).

---

## `kaggle competitions pages update`

Updates fields on an existing competition page. Only the flags you supply are
sent (the FieldMask is built from which arguments are non-default), so this is
also how you publish or unpublish a page in place.

**Usage:**

```bash
kaggle competitions pages update <competition> --page-name <current-name> \
    [-f <path>] [--new-name <name>] [--mime-type <type>] \
    [--post-title "<title>"] [--publish | --unpublish]
```

**Arguments:**

- `<competition>`: The competition slug.

**Options:**

- `--page-name <current-name>` (required): The page's current name (used as the
  identifier; rename via `--new-name`).
- `-f, --file <path>` (optional): Path to a file with the new page body.
- `--new-name <name>` (optional): Rename the page.
- `--mime-type <type>` (optional): New MIME type of the content.
- `--post-title "<title>"` (optional): New title shown above the page content.
- `--publish` / `--unpublish` (optional, mutually exclusive): Publish or
  unpublish the page.

At least one update flag is required.

**Examples:**

```bash
# Publish a staged page without changing its content.
kaggle competitions pages update my-comp --page-name rules --publish

# Swap in new content and update the visible title in one call.
kaggle competitions pages update my-comp --page-name rules \
    -f ./rules-v2.md --post-title "Competition Rules (v2)"

# Rename a page.
kaggle competitions pages update my-comp --page-name evaluation \
    --new-name scoring
```

**Note:** a small set of pages is reserved by the backend and cannot be
renamed; attempting to rename one returns an error from the server.

---

## `kaggle competitions pages delete`

Deletes a page from a competition you host. Prompts for confirmation unless
`-y/--yes` is passed (matches the existing `kaggle datasets delete` /
`kaggle kernels delete` patterns).

**Usage:**

```bash
kaggle competitions pages delete <competition> --page-name <name> [-y]
```

**Arguments:**

- `<competition>`: The competition slug.

**Options:**

- `--page-name <name>` (required): Name of the page to delete.
- `-y, --yes` (optional): Skip the confirmation prompt — useful for scripts.

**Examples:**

```bash
# Interactive: prompts "Are you sure you want to delete the page 'faq' ...?"
kaggle competitions pages delete my-comp --page-name faq

# Scripted: skip the prompt.
kaggle competitions pages delete my-comp --page-name faq -y
```

**Note:** a small set of pages is protected by the backend and cannot be
deleted; attempting to delete one returns an error from the server.

Deletion is not recoverable — there is no "undelete". List pages first with
`kaggle competitions pages list <competition>` if you're unsure of the name.

---

## `kaggle competitions launch`

Launches a competition you host. Without `--at`, the competition is launched
immediately. With `--at`, the backend schedules the launch for the given UTC
instant.

**Usage:**

```bash
kaggle competitions launch <competition> [--at <ISO-8601 UTC>]
```

**Arguments:**

- `<competition>`: The competition slug.

**Options:**

- `--at <iso>`: Schedule launch for a future UTC time. Accepts ISO-8601
  (e.g. `2027-01-01T00:00:00Z` or `2027-01-01T00:00:00+00:00`). The competition
  is launched immediately if omitted.

**Examples:**

```bash
# Launch right now.
kaggle competitions launch my-comp

# Schedule the launch for midnight UTC on 2027-01-01.
kaggle competitions launch my-comp --at 2027-01-01T00:00:00Z
```

A competition can only be launched once. Subsequent calls will be rejected by
the backend.

---

## `kaggle competitions data update`

Creates a new version of the data files for a competition you host. Uploads
via the standard blob-upload pipeline, then sends a single request bundling
the uploaded tokens. Each update **replaces the prior version's file set in
full** — there is no per-file "keep from previous" mode in v1, so list every
file you want in the new version.

**Usage:**

```bash
kaggle competitions data update <competition> -p <path> -m "<version notes>" \
    [--rerun] [--include-hidden]
```

**Arguments:**

- `<competition>`: The competition slug.

**Options:**

- `-p, --path <path>` (required): Either a **directory** (walked recursively —
  every file becomes an upload with its relative path preserved in the API's
  `name` field, e.g. `train/images/img1.jpg`), or a **single archive file**
  (e.g. a pre-packed `.zip` or `.tar`) uploaded as-is. Sub-directories are
  always traversed; hidden entries (see `--include-hidden`) are the only files
  skipped by default.
- `-m, --message "<notes>"` (required): Notes describing this version
  (e.g. `"Added test set"`).
- `--rerun` (optional): Update the RERUN databundle — the private host-only
  data swapped in during rerun scoring. Requires Kaggle admin access for now.
  Without this flag, the update targets the PUBLIC databundle (what
  participants download).
- `--include-hidden` (optional): Upload hidden files and traverse hidden
  sub-directories (names starting with `.` — e.g. `.DS_Store`, `.git/`,
  `.gitignore`). Skipped by default so you don't accidentally publish OS
  metadata or version-control detritus.

**Examples:**

```bash
# Update using a directory tree (recurses into sub-folders).
kaggle competitions data update my-comp -p ./data -m "Initial release"

# Update using a pre-packed archive as a single file (useful when you already
# need a zip for other purposes, or for directory-shaped file formats like
# Zarr).
kaggle competitions data update my-comp -p ./data.zip -m "Initial release"

# New version with a bug-fix.
kaggle competitions data update my-comp -p ./data -m "Fix label encoding in train.csv"

# Update the private rerun-scoring data.
kaggle competitions data update my-comp -p ./rerun-data \
    -m "Held-out test set" --rerun
```

**A note on directory-shaped file formats:** some formats (Zarr, some
TensorFlow SavedModel layouts, etc.) are on-disk directories that are logically
a single unit. If you pass a directory containing such a format, the recursive
walk uploads each internal chunk as its own file — often what you want for
Zarr, since participants can then stream individual chunks. If you'd rather
keep the format as an opaque single upload, pre-pack it into a `.zip` or
`.tar` and pass that file to `-p` instead.

The command prints the public URL plus the new `databundle_id` and
`databundle_version_id` on success.
