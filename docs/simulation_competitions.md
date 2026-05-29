# Tutorial: Simulation Competitions

This tutorial walks you through interacting with a Kaggle simulation competition using the CLI — from finding the competition to downloading episode replays and agent logs.

Simulation competitions (e.g., [Connect X](https://www.kaggle.com/competitions/connectx), [Lux AI](https://www.kaggle.com/competitions/lux-ai-season-3)) differ from standard competitions. Instead of submitting a CSV of predictions, you submit an agent (code) that plays against other agents in episodes. Each episode contains multiple agents competing against each other. You can identify simulation competitions on the [competitions page](https://www.kaggle.com/competitions) by their "Simulation" tag, or by looking for competitions that mention agents, bots, or game environments in their description.

## 1. Find and Inspect the Competition

Search for simulation competitions by keyword:

```bash
kaggle competitions list -s simulation
```

Once you've identified a competition (e.g., `connectx`), view its pages to read the rules, evaluation criteria, and other details:

```bash
kaggle competitions pages connectx
```

This lists the available pages (e.g., `description`, `rules`, `evaluation`, `data-description`). To read the full content of a page:

```bash
kaggle competitions pages connectx --content
```

You can also browse the competition's discussion forum to see what other participants are talking about — top strategies, common pitfalls, environment quirks. List the topics with:

```bash
kaggle competitions topics list connectx
```

This prints a table of topics with `id`, `title`, `authorName`, `commentCount`, `votes`, and `postDate`. Sort and paginate with `-s/--sort-by` (one of `hot`, `top`, `new`, `recent`, `active`, `relevance`) and `--page-size`:

```bash
kaggle competitions topics list connectx -s top --page-size 10
```

To read the full discussion under a topic, use the `show` subcommand:

```bash
kaggle competitions topics show connectx 12345
```

This returns the topic content and all its comments rendered in an indented tree structure.
```

## 2. Accept the Competition Rules

Before you can submit or download data, you **must** accept the competition rules on the Kaggle website. Navigate to the competition page (e.g., `https://www.kaggle.com/competitions/connectx`) and click "Join Competition" or "I Understand and Accept".

You can verify you've joined by checking your entered competitions:

```bash
kaggle competitions list --group entered
```

## 3. Download Competition Data

Download the competition's starter kit and any provided data:

```bash
kaggle competitions download connectx -p connectx-data
```

## 4. Submit Your Agent

Simulation competitions require you to submit agent code. You can upload files directly from your local machine.

**Single file agent** — if your agent is a single `main.py`:

```bash
kaggle competitions submit connectx -f main.py -m "Single file agent v1"
```

**Multi-file agent** — if your agent spans multiple files, bundle them into a `submission.tar.gz` with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit connectx -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission** — alternatively, you can submit via an existing Kaggle notebook:

```bash
kaggle competitions submit connectx -k YOUR_USERNAME/connectx-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## 5. Monitor Your Submission

Check the status of your submissions:

```bash
kaggle competitions submissions connectx
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
kaggle competitions replay 98765432
```

This downloads the replay JSON to your current directory as `episode-98765432-replay.json`. To specify a download location:

```bash
kaggle competitions replay 98765432 -p ./replays
```

## 8. Download Agent Logs

To debug your agent's behavior, download the logs for a specific agent in an episode. You need the episode ID and the agent's index (0-based):

```bash
# Download logs for the first agent (index 0)
kaggle competitions logs 98765432 0

# Download logs for the second agent (index 1)
kaggle competitions logs 98765432 1 -p ./logs
```

This downloads the log file as `episode-98765432-agent-0-logs.json`.

## 9. Inspect Top Teams' Active Agents

You can study how the leading teams' agents are performing — useful for scouting strategies or understanding the metagame. Start from the leaderboard to grab the team ID:

```bash
kaggle competitions leaderboard connectx -s
```

This prints a table with columns `teamId`, `teamName`, `submissionDate`, `score`. Take the `teamId` of the team you want to inspect (e.g., first place), then list every active submission they have on the leaderboard:

```bash
kaggle competitions team-submissions 42
```

This returns the team's public-safe submissions — `id`, `dateSubmitted`, and `publicScore`. For simulation competitions every leaderboard-eligible submission is listed (not just the best one), so you can see the full rotation of agents a top team is fielding.

Pick the submission with the highest `publicScore` and list its episodes, just like you would for your own:

```bash
kaggle competitions episodes 98765432
```

From there you can pull replays and agent logs for any episode that submission played in (`kaggle competitions replay <episode_id>` / `kaggle competitions logs <episode_id> <agent_index>`).

## Putting It All Together

Here's a typical workflow for iterating on a simulation competition agent:

```bash
# Download competition data
kaggle competitions download connectx -p connectx-data

# Skim discussion topics for tips before iterating
kaggle competitions topics connectx -s top
kaggle competitions topic-messages connectx <topic-id>

# Submit your agent (single file)
kaggle competitions submit connectx -f main.py -m "v1"

# Check submission status
kaggle competitions submissions connectx

# List episodes (replace with your submission ID)
kaggle competitions episodes 12345678

# Download replay and logs for an episode
kaggle competitions replay 98765432
kaggle competitions logs 98765432 0

# Check the leaderboard
kaggle competitions leaderboard connectx -s

# Scout the leader: list their active agents, then pick the best one's episodes
kaggle competitions team-submissions <leader-team-id>
kaggle competitions episodes <best-submission-id>
```
