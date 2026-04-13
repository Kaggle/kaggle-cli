# Tutorial: Simulation Competitions

This tutorial walks you through interacting with a Kaggle simulation competition using the CLI — from finding the competition to downloading episode replays and agent logs.

Simulation competitions (e.g., [Lux AI](https://www.kaggle.com/competitions/lux-ai-season-3), [Santa](https://www.kaggle.com/competitions/santa-2024)) differ from standard competitions. Instead of submitting a CSV of predictions, you submit an agent (code) that plays against other agents in episodes. Each episode contains multiple agents competing against each other.

## 1. Find and Inspect the Competition

List available competitions and look for simulation competitions:

```bash
kaggle competitions list --sort-by latestDeadline
```

Once you've identified a competition (e.g., `lux-ai-season-3`), view its pages to read the rules, evaluation criteria, and other details:

```bash
kaggle competitions pages lux-ai-season-3
```

This lists the available pages (e.g., `description`, `rules`, `evaluation`, `data-description`). To read the full content of a page:

```bash
kaggle competitions pages lux-ai-season-3 --content
```

## 2. Accept the Competition Rules

Before you can submit or download data, you **must** accept the competition rules on the Kaggle website. Navigate to the competition page (e.g., `https://www.kaggle.com/competitions/lux-ai-season-3`) and click "Join Competition" or "I Understand and Accept".

You can verify you've joined by checking your entered competitions:

```bash
kaggle competitions list --group entered
```

## 3. Download Competition Data

Download the competition's starter kit and any provided data:

```bash
mkdir lux-ai
cd lux-ai
kaggle competitions download lux-ai-season-3
```

## 4. Submit Your Agent

Simulation competitions use code submissions. First, create and push a notebook with your agent code, then submit it:

```bash
kaggle competitions submit lux-ai-season-3 -k YOUR_USERNAME/lux-ai-agent -f submission.tar.gz -v 1 -m "First agent submission"
```

## 5. Monitor Your Submission

Check the status of your submissions:

```bash
kaggle competitions submissions lux-ai-season-3
```

Note the submission ID from the output — you'll need it to view episodes.

## 6. List Episodes for a Submission

Once your submission has played some games, list the episodes:

```bash
kaggle competitions episodes 12345678
```

Replace `12345678` with your submission ID. This shows a table of episodes with columns: `id`, `createTime`, `endTime`, `state`, and `type`.

To get the output in CSV format for scripting:

```bash
kaggle competitions episodes 12345678 -v
```

## 7. Download an Episode Replay

To download the replay data for a specific episode (useful for visualizing what happened):

```bash
kaggle competitions episode-replay 98765432
```

This downloads the replay JSON to your current directory as `episode-98765432-replay.json`. To specify a download location:

```bash
kaggle competitions episode-replay 98765432 -p ./replays
```

## 8. Download Agent Logs

To debug your agent's behavior, download the logs for a specific agent in an episode. You need the episode ID and the agent's index (0-based):

```bash
# Download logs for the first agent (index 0)
kaggle competitions episode-logs 98765432 0

# Download logs for the second agent (index 1)
kaggle competitions episode-logs 98765432 1 -p ./logs
```

This downloads the log file as `episode-98765432-agent-0-logs.json`.

## Putting It All Together

Here's a typical workflow for iterating on a simulation competition agent:

```bash
# Set up
mkdir my-sim-comp && cd my-sim-comp
kaggle competitions download lux-ai-season-3

# Submit your agent
kaggle competitions submit lux-ai-season-3 -k YOUR_USERNAME/my-agent -f submission.tar.gz -v 1 -m "v1"

# Check submission status
kaggle competitions submissions lux-ai-season-3

# List episodes (replace with your submission ID)
kaggle competitions episodes 12345678

# Download replay and logs for an episode
kaggle competitions episode-replay 98765432
kaggle competitions episode-logs 98765432 0

# Check the leaderboard
kaggle competitions leaderboard lux-ai-season-3 -s
```
