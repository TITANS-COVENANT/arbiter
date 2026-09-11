"""Configuration validation and the command line.

Config validation gets its own tests because the failure mode it prevents is
expensive: a suite that runs happily against the wrong target, or with a
replicate cap that makes flagging impossible, wastes a full CI run and reports
green while doing it.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.config import (
    StatsConfig,
    SuiteConfig,
    TargetConfig,
    TaskConfig,
    load_suite,
)

runner = CliRunner()


def minimal_suite_dict(**overrides) -> dict:
    base = {
        "name": "demo",
        "tasks": [
            {
                "id": "ok",
                "input": {"baseline_rate": 0.9, "candidate_rate": 0.9, "coupling": 0.7},
            },
            {
                "id": "bad",
                "input": {"baseline_rate": 0.9, "candidate_rate": 0.5, "coupling": 0.7},
            },
        ],
        "baseline": {"kind": "python", "ref": "arbiter.sim.agent:baseline"},
        "candidate": {"kind": "python", "ref": "arbiter.sim.agent:candidate"},
        "stats": {"max_replicates": 60, "min_replicates": 4, "correction": "none"},
        "budget": {"batch_size": 16},
    }
    base.update(overrides)
    return base


@pytest.fixture
def suite_file(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(minimal_suite_dict()), encoding="utf-8")
    return path


class TestTargetValidation:
    def test_python_target_needs_a_ref(self):
        with pytest.raises(ValueError, match="requires 'ref'"):
            TargetConfig(kind="python")

    def test_subprocess_target_needs_a_command(self):
        with pytest.raises(ValueError, match="requires 'command'"):
            TargetConfig(kind="subprocess")

    def test_http_target_needs_a_url(self):
        with pytest.raises(ValueError, match="requires 'url'"):
            TargetConfig(kind="http")

    def test_concurrency_must_be_positive(self):
        with pytest.raises(ValueError, match="concurrency"):
            TargetConfig(kind="python", ref="a:b", concurrency=0)


class TestStatsValidation:
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, -0.1])
    def test_alpha_must_be_a_sensible_error_rate(self, alpha):
        with pytest.raises(ValueError, match="must lie in"):
            StatsConfig(alpha=alpha)

    def test_mde_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="mde"):
            StatsConfig(mde=1.5)

    def test_min_cannot_exceed_max(self):
        with pytest.raises(ValueError, match="min_replicates"):
            StatsConfig(min_replicates=50, max_replicates=10)

    def test_defaults_are_coherent(self):
        stats = StatsConfig()
        assert stats.min_replicates < stats.max_replicates
        assert 0 < stats.alpha < 0.5
        assert stats.correction == "e-bh"


class TestSuiteValidation:
    def test_needs_at_least_one_task(self):
        with pytest.raises(ValueError, match="at least one task"):
            SuiteConfig(
                tasks=[],
                baseline=TargetConfig(ref="a:b"),
                candidate=TargetConfig(ref="a:c"),
            )

    def test_rejects_duplicate_task_ids(self):
        with pytest.raises(ValueError, match="duplicate task ids"):
            SuiteConfig(
                tasks=[TaskConfig(id="x"), TaskConfig(id="x")],
                baseline=TargetConfig(ref="a:b"),
                candidate=TargetConfig(ref="a:c"),
            )

    def test_loads_from_yaml(self, suite_file):
        cfg = load_suite(suite_file)
        assert cfg.name == "demo"
        assert len(cfg.tasks) == 2

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="suite file not found"):
            load_suite(tmp_path / "nope.yaml")

    def test_a_yaml_list_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping at the top level"):
            load_suite(path)


class TestCli:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "arbiter" in result.stdout

    def test_help_lists_the_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("gate", "plan", "diff", "simulate", "init", "history"):
            assert command in result.stdout

    def test_init_writes_a_loadable_suite(self, tmp_path):
        path = tmp_path / "arbiter.yaml"
        result = runner.invoke(app, ["init", str(path)])
        assert result.exit_code == 0
        assert load_suite(path).name == "my-agent"

    def test_init_refuses_to_clobber(self, tmp_path):
        path = tmp_path / "arbiter.yaml"
        path.write_text("name: existing\n", encoding="utf-8")
        assert runner.invoke(app, ["init", str(path)]).exit_code == 1
        assert runner.invoke(app, ["init", str(path), "--force"]).exit_code == 0

    def test_plan_reports_both_sizings(self):
        result = runner.invoke(app, ["plan", "--tasks", "50", "--cost", "0.02"])
        assert result.exit_code == 0
        assert "fixed sample" in result.stdout
        assert "max_replicates" in result.stdout

    def test_gate_fails_on_a_regressed_suite(self, suite_file, tmp_path):
        result = runner.invoke(
            app,
            [
                "gate", str(suite_file),
                "--store", str(tmp_path / "runs.sqlite"),
                "--json", str(tmp_path / "out.json"),
                "--markdown", str(tmp_path / "out.md"),
                "--junit", str(tmp_path / "out.xml"),
                "--quiet",
            ],
        )
        assert result.exit_code == 1
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert payload["verdict"] == "fail"
        assert payload["n_flagged"] >= 1
        assert "bad" in (tmp_path / "out.md").read_text(encoding="utf-8")
        assert "<testsuite" in (tmp_path / "out.xml").read_text(encoding="utf-8")

    def test_gate_reports_a_bad_config_without_a_traceback(self, tmp_path):
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump({"tasks": []}), encoding="utf-8")
        result = runner.invoke(app, ["gate", str(path)])
        assert result.exit_code == 2

    def test_verbose_lists_every_task(self, suite_file, tmp_path):
        result = runner.invoke(
            app, ["gate", str(suite_file), "--store", str(tmp_path / "r.sqlite"), "--verbose"]
        )
        assert "All tasks" in result.stdout

    def test_diff_needs_stored_runs(self, suite_file, tmp_path):
        result = runner.invoke(
            app, ["diff", str(suite_file), "bad", "--store", str(tmp_path / "empty.sqlite")]
        )
        assert result.exit_code == 1

    def test_diff_after_a_gate_run(self, suite_file, tmp_path):
        store = str(tmp_path / "runs.sqlite")
        runner.invoke(app, ["gate", str(suite_file), "--store", store, "--quiet"])
        result = runner.invoke(app, ["diff", str(suite_file), "bad", "--store", store])
        assert result.exit_code == 0
        assert "bad" in result.stdout

    def test_diff_markdown_output(self, suite_file, tmp_path):
        store = str(tmp_path / "runs.sqlite")
        runner.invoke(app, ["gate", str(suite_file), "--store", store, "--quiet"])
        result = runner.invoke(
            app, ["diff", str(suite_file), "bad", "--store", store, "--markdown"]
        )
        assert result.exit_code == 0
        assert result.stdout.startswith("### `bad`")

    def test_history_is_empty_before_any_run(self, suite_file, tmp_path):
        result = runner.invoke(
            app, ["history", str(suite_file), "--store", str(tmp_path / "r.sqlite")]
        )
        assert "no gate runs recorded yet" in result.stdout

    def test_history_after_a_run(self, suite_file, tmp_path):
        store = str(tmp_path / "runs.sqlite")
        runner.invoke(app, ["gate", str(suite_file), "--store", store, "--quiet"])
        result = runner.invoke(app, ["history", str(suite_file), "--store", store])
        assert result.exit_code == 0
        assert "fail" in result.stdout

    def test_second_gate_run_reuses_the_baseline(self, suite_file, tmp_path):
        store = str(tmp_path / "runs.sqlite")
        runner.invoke(app, ["gate", str(suite_file), "--store", store, "--quiet"])
        second = tmp_path / "second.json"
        runner.invoke(
            app,
            ["gate", str(suite_file), "--store", store, "--json", str(second), "--quiet"],
        )
        payload = json.loads(second.read_text(encoding="utf-8"))
        assert payload["replicates_reused"] > 0

    def test_no_store_skips_the_cache(self, suite_file, tmp_path):
        out = tmp_path / "out.json"
        runner.invoke(
            app, ["gate", str(suite_file), "--no-store", "--json", str(out), "--quiet"]
        )
        assert json.loads(out.read_text(encoding="utf-8"))["replicates_reused"] == 0

    def test_simulate_reports_the_error_rates(self, tmp_path):
        out = tmp_path / "sim.json"
        result = runner.invoke(
            app,
            [
                "simulate", "--trials", "3", "--tasks", "6", "--regressed", "1",
                "--max-replicates", "60", "--json", str(out),
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["n_trials"] == 3
        assert 0.0 <= payload["fdr"] <= 1.0
