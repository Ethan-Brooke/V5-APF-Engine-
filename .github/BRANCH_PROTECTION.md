# Branch Protection Setup

These settings must be configured manually in the GitHub web UI.
Go to **Settings > Branches > Add branch protection rule** for `main`.

## Required Settings

| Setting | Value | Why |
|---------|-------|-----|
| **Branch name pattern** | `main` | Protect the default branch |
| **Require a pull request before merging** | ON | No direct pushes to main |
| **Required approvals** | 1 | You (the owner) must approve every PR |
| **Dismiss stale PR approvals when new commits are pushed** | ON | Re-review after changes |
| **Require status checks to pass before merging** | ON | CI must be green |
| **Status checks that are required** | `Run 129 Theorems (3.12)` | At minimum, latest Python must pass |
| **Require branches to be up to date before merging** | ON | No stale merges |
| **Restrict who can push to matching branches** | ON, add only yourself | Only you can push to main |
| **Allow force pushes** | OFF | Protect commit history |
| **Allow deletions** | OFF | Prevent branch deletion |

## What This Achieves

- **AI tools can**: fork, clone, open PRs, comment, run CI checks, read all code.
- **AI tools cannot**: merge PRs, push to main, delete branches, or bypass CI.
- **Only you** can approve and merge PRs into main.

## Optional: Allow Claude Code Write Access

If you want Claude Code (via claude.ai/code or the CLI) to open PRs on your behalf:

1. Go to **Settings > Collaborators** and add the Claude GitHub app (if using Claude Code web).
2. Grant it **write** access (needed to create branches and PRs).
3. Branch protection still prevents direct merges — you remain the gatekeeper.
