# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations. Target `oldyi317/TVL-Analysis` explicitly with `--repo oldyi317/TVL-Analysis` (or `repos/oldyi317/TVL-Analysis/...` for `gh api`), because this clone also has a private remote.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. For multi-line bodies, write the exact text to a temporary file and use `--body-file <path>`.
- **Read an issue**: `gh issue view <number> --json number,title,body,labels,comments`.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Do not rely on automatic remote selection; use the explicit target above.

## Repo quirk: GraphQL `projectCards` is deprecated here

`gh issue view <n>` and `gh pr view <n>` without `--json`, and `gh pr edit`, fail on this repo with `GraphQL: Projects (classic) is being deprecated ... projectCards`. Use the REST or field-scoped forms instead:

- **Read**: `gh issue view <n> --json title,body,comments` / `gh pr view <n> --json title,body,commits`
- **Edit a PR body**: `gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -F body=@<file>`
- **Inspect a PR**: `gh api repos/<owner>/<repo>/pulls/<n> --jq '.commits, .body'`

`gh issue create`, `gh pr create`, `gh issue list --json`, `gh pr list --json`, `gh issue comment` and `gh issue close` work as-is.

### Timeout variant (WSL)

`gh issue view` / `gh issue create` use GraphQL, which can time out in WSL (`Post "https://api.github.com/graphql": dial tcp ...:443: i/o timeout`) while REST `gh api` still works. After one timeout in a session, switch fully to `gh api` and do not retry the GraphQL forms:

- **Read body**: `gh api repos/<o>/<r>/issues/<n>`; **comments**: `.../issues/<n>/comments`
- **Create issue**: `gh api --method POST repos/<o>/<r>/issues -f title=... -f body=... -f 'labels[]=...'`
- **Dependencies**: `gh api .../issues/<n>/dependencies/blocked_by`

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either — resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

A spec with child tickets (or a set of children under an existing parent): write the drafts into one directory and run `bash scripts/publish-tickets.sh <dir>`. It creates the issues, replaces `{{SPEC}}` / `{{<id>}}` placeholders with real numbers, links each child as a sub-issue, adds the blocking edges, and reads the result back. The directory layout is in the script header; `DRY_RUN=1` prints what it would do.

Before publishing, run `python tests/test_vocabulary.py <draft files>` so the drafts carry no avoided glossary terms.

The two calls the script wraps, for one-off use:

- **Sub-issue**: `gh api --method POST repos/oldyi317/TVL-Analysis/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>`
- **Blocking edge**: `gh api --method POST repos/oldyi317/TVL-Analysis/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`

Both take the numeric **database id** (`gh api repos/oldyi317/TVL-Analysis/issues/<n> --jq .id`), not the `#number`.

A ticket is unblocked only when its blockers are **closed**. Check that each ticket can close without an action that belongs to a ticket it blocks (a PR that is prepared in one ticket and merged in the next must not write `Closes`), and that every value handed across tickets names who produces it, who commits it, and who reads it.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo oldyi317/TVL-Analysis --json number,title,body,labels,comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies** — the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only — the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
