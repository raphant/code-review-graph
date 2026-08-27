---
name: code-review-graph
description: Operate code-review-graph (CRG) — query callers, tests, blast radius and architecture from a repo's structural graph, keep that graph and its embeddings fresh, and wire CRG into a repository's MCP clients. Read it before you trust a graph answer, because a zero result and a `head_matches_build: true` each lie in a way named here. Use on CRG, graph tools, blast radius, semantic code search, a stale graph, or `.code-review-graph`.
---

# code-review-graph

CRG narrows source reads with a local structural graph. Use it for relationships and
multi-hop impact. Use exact source for the final conclusion.

## Prove a zero

CRG reports absence in three places, and all three produce a convincing false zero. Every
one of them needs a **positive control**: a query you know returns rows, run the same way,
before you believe the empty one.

| Zero | What it can really mean |
| --- | --- |
| `search_mode: none` and no results | The phrasing fell through every strategy |
| An empty `diff` of the installed copy | The `find` that fed it returned nothing |
| `pgrep -f "code-review-graph watch"` finds no watchers | macOS matched a truncated command line |

## Use the graph

1. Call `get_minimal_context_tool(task="<task>")` first.
2. Find the entity with `semantic_search_nodes_tool` or a specific graph query.
3. Use `query_graph_tool` for `callers_of`, `callees_of`, `imports_of`, `importers_of`,
   `tests_for`, or `children_of`.
4. Use `get_impact_radius_tool`, `detect_changes_tool`, or `get_affected_flows_tool` for
   change risk.
5. Use `get_architecture_overview_tool` and the community tools for system shape.
6. Open the implementation and its tests, and read them.

You are done when the claim rests on source you read, not on a graph row alone.

Prefer `detail_level="minimal"`, and escalate only when the compact result lacks evidence.
Common symbol names are ambiguous, so rerun with the returned `qualified_name`.

Across several registered repos, start with `list_repos_tool` or `cross_repo_search_tool`.
They read the registry and take no `repo_root`. Every other per-repo tool needs
`repo_root` as an absolute path whenever the working directory is not a project root.

**Read `search_mode` before you read the results.** It is `hybrid` (FTS plus vectors),
`semantic` (vectors only), `fts`, `keyword` (LIKE fallback), or `none`, and `confidence`
explains a thin result in the same payload. A whole-sentence query reliably lands on
`none`. Treat that as "this phrasing found nothing", and retry with identifiers or one or
two words. Establish absence with a **positive control**.

## Keep the graph fresh

| Situation | Command |
| --- | --- |
| Normal source edits | hooks or plugin run an incremental update |
| Checkout, pull, rebase, branch move | `code-review-graph update --repo <repo>` |
| Large drift, parser upgrade, doubtful data | `code-review-graph build --repo <repo>` |
| Refresh plus change report | `code-review-graph update --brief --repo <repo>` |
| Read-only change report | `code-review-graph detect-changes --brief --repo <repo>` |

Hooks catch file edits, but a moving Git HEAD still leaves the graph stale. Compare
`status`'s built commit with `git -C <repo> rev-parse HEAD` after any Git operation. MCP
results carry the same check as `_graph.head_matches_build`.

**A matching commit is necessary, not sufficient.** Both checks compare commits, so
neither sees uncommitted work: a dirty tree reports `head_matches_build: true` while the
graph still holds the pre-edit source. Run `update` whenever `git status --porcelain` is
non-empty. Freshness is proven when the commit matches *and* the tree is clean or has just
been updated.

`status` also warns that the graph was built on a different *branch*. When the built commit
still equals HEAD, that warning is about the branch label alone, and `update` re-stamps it
in seconds.

## Keep embeddings current

A graph has no embeddings until you build them. Until then, search covers names, paths,
signatures, and FTS data — not source text or meaning.

```bash
uv tool install --force '.[embeddings]'          # from the source; adds sentence-transformers
code-review-graph embed --repo <repo>            # provider defaults to local
```

**Always reinstall with `'.[embeddings]'`.** `sentence-transformers` lives only in that
extra, so a plain `uv tool install --force .` strips it, and nothing announces the loss:
`embed` prints an error, but every daemon watcher only logs a refresh failure and keeps
running. Semantic search then decays across all repos while the daemon reports healthy.

Use the `local` provider on private code. The `openai`, `google`, `minimax`, and `voyage`
providers send source text to a third party.

The local default `all-MiniLM-L6-v2` (384 dims) is pulled from HuggingFace on first run and
cached under `~/.cache/huggingface/hub`. Once it is cached, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` so a run never waits on the network. Encoding uses MPS on Apple
Silicon: 61k nodes across eight repos took about 86 seconds.

**A service-managed daemon does not inherit your shell.** Put those two variables in the
service definition — `EnvironmentVariables` in a launchd plist, `Environment=` under
systemd. Without them, a watcher refresh behind a TLS-inspecting proxy fails quietly.
Confirm with `grep -c CERTIFICATE ~/.code-review-graph/logs/*.log`, which should read 0.

**Embeddings only stay fresh if the daemon is told to refresh them.** `embed` is a one-shot
pass and `update` never embeds. The daemon does, but only when `~/.code-review-graph/watch.toml`
carries both keys — under `[daemon]` for every repo, or per `[[repos]]`:

```toml
[daemon]
embedding_provider = "local"
embedding_model = "all-MiniLM-L6-v2"
```

`crg-daemon add <path> --embedding-provider local --embedding-model all-MiniLM-L6-v2` does
the same. Supply both keys; half a pair is dropped with a warning. Refresh is refresh-only,
so run `embed` once before the daemon can maintain anything.

Without the pair, every node added since the last `embed` is unembedded, and semantic
search quietly degrades toward FTS on exactly the newest code. Measure that rather than
guess: `crg-daemon status` shows the resolved provider per repo under `Embed` and names the
repos still showing `-`, and `list_graph_stats_tool` returns `unembedded_count` beside
`embeddings_count`. Read the two together, because a healthy-looking count still hides a gap.

**One decay hole stays open.** A node whose body changed keeps its qualified name and still
counts as embedded even when its vector is stale. Only a running daemon refresh closes that
half.

A failed refresh (rate limit, offline provider, missing key) leaves the graph correct with
stale vectors and logs "kept the graph current but not its embeddings". The watcher lives.
A backlog catches up on the next watch update, because the refresh walks the whole graph
rather than only the changed files.

## Install in a repository

Read the repo's own guidance and inspect dirty files first. Preserve every unrelated MCP
server, hook, instruction, and worktree change.

```bash
code-review-graph install --platform opencode --repo <repo> --yes
code-review-graph install --platform claude-code --repo <repo> --yes
```

`--dry-run` previews repo files only, so its output understates the blast radius by
omitting the user-level writes. `--platform opencode` and `all` overwrite
`~/.config/opencode/plugins/crg-plugin.ts`, which every OpenCode project on the machine
loads; `--platform all` can also write Cursor hooks, Gemini CLI settings, and a git
pre-commit hook. Pass `--no-hooks` to keep the install inside the repo.

Then inspect the generated configs and replace CRG's command with an absolute path.

```json
// opencode.json, opencode.jsonc, or .opencode/opencode.json
"code-review-graph": {
  "type": "local",
  "command": ["<abs-path>/code-review-graph", "serve", "--repo", "<abs-repo-path>"]
}

// .mcp.json
"code-review-graph": {
  "command": "<abs-path>/code-review-graph",
  "args": ["serve"],
  "cwd": "<abs-repo-path>",
  "type": "stdio"
}
```

**Pin that absolute path.** An installer-generated `uvx code-review-graph` fetches the
published package, so a locally built or patched install is bypassed with no warning, and
`--version` reports the same number either way. `uv tool install` also copies a snapshot of
the **working tree**, not of git, so commits are irrelevant and edits made since the last
install are absent. Compare the installed copy with the source, guarding the `find` so an
empty result cannot pass as silence:

```bash
INST=$(find ~/.local/share/uv/tools/code-review-graph/lib -maxdepth 4 -type d -name code_review_graph | head -1)
[ -n "$INST" ] || echo "NOT FOUND — the diff below would be a false pass"
diff -rq -x '__pycache__' -x '*.pyc' code_review_graph "$INST"   # silence = identical
```

Keep `.code-review-graph/` ignored, then build:

```bash
code-review-graph build --repo <repo>
code-review-graph status --repo <repo>
```

**You are done when the graph exists, its built commit matches `git rev-parse HEAD`, and
every client the repo actually configures has completed a real tool call.** Check only the
clients the repo configures: a repo carrying just `.mcp.json` needs just the Claude Code
check. A health list proves connection; completion needs a tool record.

- **OpenCode.** `opencode mcp list` shows CRG `connected`. Then run an unprimed smoke
  request and require a `tool_use` event for
  `code-review-graph_list_graph_stats_tool` whose `state.status` is `completed`:

  ```bash
  opencode run --format json \
    "Invoke code-review-graph list_graph_stats_tool once. Return only files_count."
  ```

- **Claude Code.** Start Claude in the repo and accept **Use this MCP server**, the
  narrower of the two approvals. CRG tools are deferred, so the smoke prompt must load and
  call one: *"Load `mcp__code-review-graph__list_graph_stats_tool` with ToolSearch, then
  invoke it once. Answer only from its tool result, with the single word VERIFIED."*
  Evidence is the `tool_use` and matching `tool_result` in the persisted transcript, not
  the word VERIFIED.

**A client keeps the server it launched with.** After you edit a config, the running
session still holds the old command, so a passing tool call there proves the graph rather
than the new config. Restart the client before you call a config verified.

This outlives your own session: every other open client holds its own `serve` process from
whenever it started, running the old code until restarted. List them and compare against
the install time, guarding the match so the command does not find itself:

```bash
ps -A -o pid=,lstart=,args= | awk '/code-review-graph serve/ && !/awk|grep/'
```

Report which sessions need a restart, and leave those processes running. Killing one breaks
the graph tools in a session someone may be mid-task in.

## Remove CRG

Preview first, then remove only CRG-owned entries and files:

```bash
code-review-graph uninstall --dry-run --repo <repo>
code-review-graph uninstall --yes --repo <repo>
```

Use `--keep-data` to remove the client integrations but keep the graph database.
