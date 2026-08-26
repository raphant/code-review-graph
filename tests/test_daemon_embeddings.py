"""Tests for daemon-driven embedding refresh.

Covers the decay gap: a watcher that keeps the graph current but never its
vectors, so semantic search silently degrades on the newest code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_review_graph.daemon import (
    DaemonConfig,
    WatchDaemon,
    WatchRepo,
    add_repo_to_config,
    load_config,
    save_config,
)

LOCAL_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path, name: str) -> Path:
    """Create a directory that passes the daemon's repo-marker check."""
    repo = tmp_path / name
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def _write_config(tmp_path: Path, body: str) -> Path:
    config = tmp_path / "watch.toml"
    config.write_text(body, encoding="utf-8")
    return config


def _daemon_section(tmp_path: Path, extra: str = "") -> str:
    return (
        "[daemon]\n"
        'session_name = "test-session"\n'
        f'log_dir = "{(tmp_path / "logs").as_posix()}"\n'
        "poll_interval = 5\n"
        f"{extra}"
    )


def _repo_section(repo: Path, alias: str, extra: str = "") -> str:
    return (
        "\n[[repos]]\n"
        f'path = "{repo.as_posix()}"\n'
        f'alias = "{alias}"\n'
        f"{extra}"
    )


# ---------------------------------------------------------------------------
# G1 / G2: watch.toml config
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_config_reads_daemon_default_and_repo_override(self, tmp_path):
        plain = _repo(tmp_path, "plain")
        custom = _repo(tmp_path, "custom")
        config_file = _write_config(
            tmp_path,
            _daemon_section(
                tmp_path,
                'embedding_provider = "local"\n'
                f'embedding_model = "{LOCAL_MODEL}"\n',
            )
            + _repo_section(plain, "plain")
            + _repo_section(
                custom,
                "custom",
                'embedding_provider = "voyage"\nembedding_model = "voyage-code-3"\n',
            ),
        )

        cfg = load_config(config_file)

        assert cfg.embedding_provider == "local"
        assert cfg.embedding_model == LOCAL_MODEL

        by_alias = {r.alias: r for r in cfg.repos}
        # A repo that names nothing keeps its own keys empty and inherits.
        assert by_alias["plain"].embedding_provider is None
        assert cfg.resolved_embedding(by_alias["plain"]) == ("local", LOCAL_MODEL)
        # A repo that names its own pair overrides the daemon default.
        assert cfg.resolved_embedding(by_alias["custom"]) == (
            "voyage",
            "voyage-code-3",
        )

    def test_config_without_embedding_keys_resolves_to_none(self, tmp_path):
        repo = _repo(tmp_path, "bare")
        config_file = _write_config(
            tmp_path, _daemon_section(tmp_path) + _repo_section(repo, "bare")
        )

        cfg = load_config(config_file)

        assert cfg.embedding_provider is None
        assert cfg.resolved_embedding(cfg.repos[0]) is None

    def test_config_never_mixes_repo_provider_with_daemon_model(self, tmp_path):
        """A repo naming its own provider must not borrow the daemon's model."""
        repo = _repo(tmp_path, "mixer")
        cfg = DaemonConfig(
            embedding_provider="local",
            embedding_model=LOCAL_MODEL,
            repos=[
                WatchRepo(
                    path=str(repo),
                    alias="mixer",
                    embedding_provider="voyage",
                    embedding_model="voyage-code-3",
                )
            ],
        )

        assert cfg.resolved_embedding(cfg.repos[0]) == ("voyage", "voyage-code-3")

    def test_config_roundtrips_through_save_and_load(self, tmp_path):
        plain = _repo(tmp_path, "rt-plain")
        custom = _repo(tmp_path, "rt-custom")
        original = DaemonConfig(
            session_name="rt",
            log_dir=tmp_path / "rt-logs",
            poll_interval=3,
            embedding_provider="local",
            embedding_model=LOCAL_MODEL,
            repos=[
                WatchRepo(path=str(plain), alias="rt-plain"),
                WatchRepo(
                    path=str(custom),
                    alias="rt-custom",
                    embedding_provider="openai",
                    embedding_model="text-embedding-3-small",
                ),
            ],
        )
        config_file = tmp_path / "rt.toml"
        save_config(original, config_file)

        reloaded = load_config(config_file)

        assert reloaded.embedding_provider == "local"
        assert reloaded.embedding_model == LOCAL_MODEL
        by_alias = {r.alias: r for r in reloaded.repos}
        assert by_alias["rt-plain"].embedding_provider is None
        assert by_alias["rt-custom"].embedding_provider == "openai"
        assert by_alias["rt-custom"].embedding_model == "text-embedding-3-small"

    @pytest.mark.parametrize(
        "extra",
        [
            'embedding_provider = "local"\n',
            f'embedding_model = "{LOCAL_MODEL}"\n',
        ],
    )
    def test_half_configured_daemon_pair_is_dropped(self, tmp_path, extra, caplog):
        repo = _repo(tmp_path, "half-daemon")
        config_file = _write_config(
            tmp_path,
            _daemon_section(tmp_path, extra) + _repo_section(repo, "half-daemon"),
        )

        cfg = load_config(config_file)

        assert cfg.embedding_provider is None
        assert cfg.embedding_model is None
        assert cfg.resolved_embedding(cfg.repos[0]) is None
        assert "embedding_provider and embedding_model" in caplog.text

    @pytest.mark.parametrize(
        "extra",
        [
            'embedding_provider = "local"\n',
            f'embedding_model = "{LOCAL_MODEL}"\n',
        ],
    )
    def test_half_configured_repo_pair_is_dropped_and_inherits(
        self, tmp_path, extra, caplog
    ):
        repo = _repo(tmp_path, "half-repo")
        config_file = _write_config(
            tmp_path,
            _daemon_section(
                tmp_path,
                'embedding_provider = "local"\n'
                f'embedding_model = "{LOCAL_MODEL}"\n',
            )
            + _repo_section(repo, "half-repo", extra),
        )

        cfg = load_config(config_file)

        assert cfg.repos[0].embedding_provider is None
        assert cfg.repos[0].embedding_model is None
        # Dropping a broken override falls back to the daemon default rather
        # than silently disabling refresh for that repo.
        assert cfg.resolved_embedding(cfg.repos[0]) == ("local", LOCAL_MODEL)
        assert "half-repo" in caplog.text

    def test_half_configured_add_repo_is_rejected(self, tmp_path):
        repo = _repo(tmp_path, "half-add")
        config_file = tmp_path / "watch.toml"

        with pytest.raises(ValueError, match="supplied together"):
            add_repo_to_config(
                str(repo),
                alias="half-add",
                config_path=config_file,
                embedding_provider="local",
            )


# ---------------------------------------------------------------------------
# G3: watcher spawn
# ---------------------------------------------------------------------------


class TestWatcherSpawn:
    @staticmethod
    def _spawn(tmp_path, config: DaemonConfig) -> list[str]:
        daemon = WatchDaemon(config=config, config_path=tmp_path / "watch.toml")
        with patch(
            "code_review_graph.daemon.subprocess.Popen",
            return_value=MagicMock(pid=4321, poll=MagicMock(return_value=None)),
        ) as popen:
            daemon._start_watcher(config.repos[0])
        return list(popen.call_args.args[0])

    def test_spawn_passes_resolved_embedding_flags(self, tmp_path):
        repo = _repo(tmp_path, "spawn-inherit")
        config = DaemonConfig(
            log_dir=tmp_path / "logs",
            embedding_provider="local",
            embedding_model=LOCAL_MODEL,
            repos=[WatchRepo(path=str(repo), alias="spawn-inherit")],
        )

        cmd = self._spawn(tmp_path, config)

        assert "watch" in cmd
        assert cmd[cmd.index("--embedding-provider") + 1] == "local"
        assert cmd[cmd.index("--embedding-model") + 1] == LOCAL_MODEL

    def test_spawn_prefers_repo_override(self, tmp_path):
        repo = _repo(tmp_path, "spawn-override")
        config = DaemonConfig(
            log_dir=tmp_path / "logs",
            embedding_provider="local",
            embedding_model=LOCAL_MODEL,
            repos=[
                WatchRepo(
                    path=str(repo),
                    alias="spawn-override",
                    embedding_provider="voyage",
                    embedding_model="voyage-code-3",
                )
            ],
        )

        cmd = self._spawn(tmp_path, config)

        assert cmd[cmd.index("--embedding-provider") + 1] == "voyage"
        assert cmd[cmd.index("--embedding-model") + 1] == "voyage-code-3"

    def test_spawn_omits_flags_when_unconfigured(self, tmp_path):
        repo = _repo(tmp_path, "spawn-none")
        config = DaemonConfig(
            log_dir=tmp_path / "logs",
            repos=[WatchRepo(path=str(repo), alias="spawn-none")],
        )

        cmd = self._spawn(tmp_path, config)

        assert "--embedding-provider" not in cmd
        assert "--embedding-model" not in cmd

    def test_spawn_flags_are_a_valid_watch_invocation(self, tmp_path):
        """The flags handed to the child must parse as real `watch` options."""
        repo = _repo(tmp_path, "spawn-parse")
        config = DaemonConfig(
            log_dir=tmp_path / "logs",
            embedding_provider="local",
            embedding_model=LOCAL_MODEL,
            repos=[WatchRepo(path=str(repo), alias="spawn-parse")],
        )
        cmd = self._spawn(tmp_path, config)

        from code_review_graph.cli import _embedding_refresh_kwargs

        watch_args = cmd[cmd.index("watch") + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--repo")
        from code_review_graph.cli import _add_embedding_refresh_args

        _add_embedding_refresh_args(parser)
        parsed = parser.parse_args(watch_args)

        assert _embedding_refresh_kwargs(parsed, parser) == {
            "embedding_provider": "local",
            "embedding_model": LOCAL_MODEL,
        }

    def test_reconcile_restarts_a_watcher_whose_embedding_changed(self, tmp_path):
        """Editing watch.toml's provider must take effect without a full restart."""
        repo = _repo(tmp_path, "reconcile")
        (repo / ".code-review-graph").mkdir()
        (repo / ".code-review-graph" / "graph.db").touch()
        before = DaemonConfig(
            log_dir=tmp_path / "logs",
            repos=[WatchRepo(path=str(repo), alias="reconcile")],
        )
        daemon = WatchDaemon(config=before, config_path=tmp_path / "watch.toml")

        with (
            patch(
                "code_review_graph.daemon.subprocess.Popen",
                return_value=MagicMock(pid=1, poll=MagicMock(return_value=None)),
            ),
            patch("code_review_graph.registry.Registry"),
            patch.object(daemon, "_save_state"),
        ):
            daemon._start_watcher(before.repos[0])
            daemon._current_repos["reconcile"] = before.repos[0]
            assert daemon._current_embeddings["reconcile"] is None

            after = DaemonConfig(
                log_dir=tmp_path / "logs",
                embedding_provider="local",
                embedding_model=LOCAL_MODEL,
                repos=[WatchRepo(path=str(repo), alias="reconcile")],
            )
            with patch.object(
                daemon, "_start_watcher", wraps=daemon._start_watcher
            ) as restart:
                daemon.reconcile(after)

        assert restart.call_count == 1
        assert daemon._current_embeddings["reconcile"] == ("local", LOCAL_MODEL)


# ---------------------------------------------------------------------------
# G4 / G7: CLI
# ---------------------------------------------------------------------------


class TestDaemonCli:
    def test_daemon_add_persists_the_pair(self, tmp_path, capsys):
        repo = _repo(tmp_path, "cli-add")
        config_file = tmp_path / "watch.toml"

        with (
            patch(
                "code_review_graph.daemon.default_config_path",
                return_value=config_file,
            ),
            patch("code_review_graph.daemon.is_daemon_running", return_value=False),
        ):
            from code_review_graph.daemon_cli import _handle_add

            _handle_add(
                argparse.Namespace(
                    path=str(repo),
                    alias="cli-add",
                    embedding_provider="local",
                    embedding_model=LOCAL_MODEL,
                )
            )

        assert "Embedding refresh: local" in capsys.readouterr().out
        cfg = load_config(config_file)
        assert cfg.repos[0].embedding_provider == "local"
        assert cfg.repos[0].embedding_model == LOCAL_MODEL

    def test_daemon_add_without_flags_leaves_config_clean(self, tmp_path):
        repo = _repo(tmp_path, "cli-add-plain")
        config_file = tmp_path / "watch.toml"

        with (
            patch(
                "code_review_graph.daemon.default_config_path",
                return_value=config_file,
            ),
            patch("code_review_graph.daemon.is_daemon_running", return_value=False),
        ):
            from code_review_graph.daemon_cli import _handle_add

            _handle_add(
                argparse.Namespace(
                    path=str(repo),
                    alias="cli-add-plain",
                    embedding_provider=None,
                    embedding_model=None,
                )
            )

        assert "embedding_provider" not in config_file.read_text(encoding="utf-8")

    def test_daemon_add_rejects_half_a_pair(self, tmp_path, capsys):
        repo = _repo(tmp_path, "cli-add-half")
        config_file = tmp_path / "watch.toml"

        with (
            patch(
                "code_review_graph.daemon.default_config_path",
                return_value=config_file,
            ),
            pytest.raises(SystemExit),
        ):
            from code_review_graph.daemon_cli import _handle_add

            _handle_add(
                argparse.Namespace(
                    path=str(repo),
                    alias="cli-add-half",
                    embedding_provider="local",
                    embedding_model=None,
                )
            )

        assert "supplied together" in capsys.readouterr().out

    def test_daemon_add_cli_flags_are_registered(self):
        """`daemon add` must actually accept the flags, not just the handler."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "code_review_graph", "daemon", "add", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert "--embedding-provider" in result.stdout
        assert "--embedding-model" in result.stdout

    def test_status_shows_embedding_identity_and_warns_when_absent(
        self, tmp_path, capsys
    ):
        embedded = _repo(tmp_path, "with-embed")
        plain = _repo(tmp_path, "without-embed")
        config = DaemonConfig(
            log_dir=tmp_path / "logs",
            repos=[
                WatchRepo(
                    path=str(embedded),
                    alias="with-embed",
                    embedding_provider="local",
                    embedding_model=LOCAL_MODEL,
                ),
                WatchRepo(path=str(plain), alias="without-embed"),
            ],
        )

        with (
            patch("code_review_graph.daemon.load_config", return_value=config),
            patch("code_review_graph.daemon.is_daemon_running", return_value=False),
        ):
            from code_review_graph.daemon_cli import _handle_status

            _handle_status(argparse.Namespace())

        out = capsys.readouterr().out
        assert "Embed" in out
        assert f"local/{LOCAL_MODEL}" in out
        # The repo with no refresh is named, not silently blank.
        assert "without-embed" in out
        assert "its vectors do not" in out

    def test_status_stays_quiet_when_every_repo_refreshes(self, tmp_path, capsys):
        repo = _repo(tmp_path, "all-embed")
        config = DaemonConfig(
            log_dir=tmp_path / "logs",
            embedding_provider="local",
            embedding_model=LOCAL_MODEL,
            repos=[WatchRepo(path=str(repo), alias="all-embed")],
        )

        with (
            patch("code_review_graph.daemon.load_config", return_value=config),
            patch("code_review_graph.daemon.is_daemon_running", return_value=False),
        ):
            from code_review_graph.daemon_cli import _handle_status

            _handle_status(argparse.Namespace())

        assert "its vectors do not" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# G5: a failing provider must not crash-loop the watcher
# ---------------------------------------------------------------------------


class TestWatchSurvivesEmbeddingFailure:
    def test_watch_survives_a_failed_embedding_refresh(self, caplog):
        from code_review_graph.incremental import _raise_watch_postprocess_warnings
        from code_review_graph.postprocessing import EMBEDDING_WARNINGS_KEY

        result = {
            "warnings": ["Embedding refresh failed: RuntimeError: rate limited"],
            EMBEDDING_WARNINGS_KEY: [
                "Embedding refresh failed: RuntimeError: rate limited"
            ],
        }

        _raise_watch_postprocess_warnings(result)  # must not raise

        assert "not its embeddings" in caplog.text

    def test_watch_still_fails_on_a_real_postprocess_warning(self):
        from code_review_graph.incremental import _raise_watch_postprocess_warnings
        from code_review_graph.postprocessing import EMBEDDING_WARNINGS_KEY

        result = {
            "warnings": [
                "Embedding refresh failed: RuntimeError: rate limited",
                "FTS rebuild failed: OperationalError: database is locked",
            ],
            EMBEDDING_WARNINGS_KEY: [
                "Embedding refresh failed: RuntimeError: rate limited"
            ],
        }

        with pytest.raises(RuntimeError, match="database is locked"):
            _raise_watch_postprocess_warnings(result)

    def test_watch_still_fails_when_nothing_is_tagged(self):
        from code_review_graph.incremental import _raise_watch_postprocess_warnings

        with pytest.raises(RuntimeError, match="flow tracing"):
            _raise_watch_postprocess_warnings({"warnings": ["flow tracing failed"]})

    def test_postprocess_tags_embedding_failures(self, tmp_path):
        from code_review_graph.graph import GraphStore
        from code_review_graph.postprocessing import (
            EMBEDDING_WARNINGS_KEY,
            run_post_processing,
        )

        store = GraphStore(str(tmp_path / "graph.db"))
        try:
            with patch(
                "code_review_graph.embeddings.refresh_embeddings",
                side_effect=RuntimeError("provider unavailable offline"),
            ):
                result = run_post_processing(
                    store,
                    embedding_provider="local",
                    embedding_model="test-model",
                )
        finally:
            store.close()

        # Duplicated, not moved: existing callers reading "warnings" still see it.
        assert any(
            "provider unavailable offline" in w for w in result["warnings"]
        )
        assert any(
            "provider unavailable offline" in w for w in result[EMBEDDING_WARNINGS_KEY]
        )


# ---------------------------------------------------------------------------
# G6: the shortfall is measurable
# ---------------------------------------------------------------------------


class _StubProvider:
    """Deterministic provider so embedding coverage is testable offline."""

    dimension = 2
    name = "local:stub"

    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _first_function(store) -> str:
    """Qualified name of one embeddable node, as the graph actually spells it."""
    row = store._conn.execute(
        "SELECT qualified_name FROM nodes WHERE kind != 'File' "
        "ORDER BY qualified_name LIMIT 1",
    ).fetchone()
    return str(row[0])


class TestUnembeddedCount:
    @staticmethod
    def _graph(tmp_path):
        """Two embeddable Function nodes plus one File node, which is not."""
        from code_review_graph.graph import GraphStore
        from code_review_graph.parser import NodeInfo

        # list_graph_stats validates repo_root, so the fixture needs a marker.
        (tmp_path / ".code-review-graph").mkdir(exist_ok=True)
        db_path = tmp_path / "graph.db"
        file_path = str(tmp_path / "mod.py")
        store = GraphStore(db_path)
        store.upsert_node(
            NodeInfo(
                kind="File",
                name=file_path,
                file_path=file_path,
                line_start=1,
                line_end=4,
                language="python",
            )
        )
        for name, start in (("alpha", 1), ("beta", 3)):
            store.upsert_node(
                NodeInfo(
                    kind="Function",
                    name=name,
                    file_path=file_path,
                    line_start=start,
                    line_end=start + 1,
                    language="python",
                )
            )
        store.commit()
        store.close()
        return db_path

    def test_unembedded_count_ignores_file_nodes(self, tmp_path):
        from code_review_graph.embeddings import EmbeddingStore

        db_path = self._graph(tmp_path)
        store = EmbeddingStore(db_path)
        try:
            # Nothing embedded yet: both functions missing, the File node is not
            # embeddable and must not inflate the shortfall.
            assert store.unembedded_count() == 2

            store._conn.execute(
                "INSERT INTO embeddings (qualified_name, vector, text_hash, provider)"
                " VALUES (?, ?, ?, ?)",
                (_first_function(store), b"\x00" * 4, "hash", "local:test"),
            )
            store._conn.commit()

            assert store.unembedded_count() == 1
            assert store.count() == 1
        finally:
            store.close()

    def test_stats_reports_and_explains_the_shortfall(self, tmp_path):
        from code_review_graph.embeddings import EmbeddingStore
        from code_review_graph.tools.query import list_graph_stats

        db_path = self._graph(tmp_path)
        store = EmbeddingStore(db_path)
        try:
            store._conn.execute(
                "INSERT INTO embeddings (qualified_name, vector, text_hash, provider)"
                " VALUES (?, ?, ?, ?)",
                (_first_function(store), b"\x00" * 4, "hash", "local:test"),
            )
            store._conn.commit()
        finally:
            store.close()

        with patch(
            "code_review_graph.tools.query.get_db_path", return_value=str(db_path)
        ):
            result = list_graph_stats(repo_root=str(tmp_path))

        assert result["embeddings_count"] == 1
        assert result["unembedded_count"] == 1
        assert "1 node(s) have no embedding" in result["summary"]
        assert "watch.toml" in result["summary"]

    def test_virtual_nodes_are_embeddable(self, tmp_path):
        """A node outside the file inventory must still be reachable by embed.

        Spring Event markers carry a real name but no File node, so a walk over
        get_all_files() left them permanently unembedded while unembedded_count
        kept reporting a shortfall no command could close.
        """
        from code_review_graph.embeddings import EmbeddingStore, embed_all_nodes
        from code_review_graph.graph import GraphStore
        from code_review_graph.parser import NodeInfo

        db_path = self._graph(tmp_path)
        graph = GraphStore(db_path)
        try:
            graph.upsert_node(
                NodeInfo(
                    kind="Event",
                    name="org.springframework.cloud.bus.event.RemoteApplicationEvent",
                    file_path="event",
                    line_start=0,
                    line_end=0,
                    language="java",
                    extra={"virtual": True, "event_type": "RemoteApplicationEvent"},
                )
            )
            graph.commit()

            # The inventory the old code walked genuinely omits it...
            assert "event" not in graph.get_all_files()
            # ...but it is a real, embeddable node.
            assert any(n.file_path == "event" for n in graph.get_all_nodes())

            store = EmbeddingStore(db_path, provider="local", model="stub")
            store.provider = _StubProvider()
            store.available = True
            try:
                embedded = embed_all_nodes(graph, store)
                assert embedded == 3  # two functions plus the virtual node
                assert store.unembedded_count() == 0
            finally:
                store.close()
        finally:
            graph.close()

    def test_stats_stays_quiet_on_a_graph_that_was_never_embedded(self, tmp_path):
        from code_review_graph.tools.query import list_graph_stats

        db_path = self._graph(tmp_path)

        with patch(
            "code_review_graph.tools.query.get_db_path", return_value=str(db_path)
        ):
            result = list_graph_stats(repo_root=str(tmp_path))

        # Never embedded is a choice, not decay — no shortfall nag.
        assert result["embeddings_count"] == 0
        assert result["unembedded_count"] == 2
        assert "have no embedding" not in result["summary"]
