#!/usr/bin/env python3
"""Read, from an agentic InferenceX row's two logs, the arguments that replay it in BLIS.

An `agentic_traces` row in InferenceX is one AgentX benchmark (`aiperf profile --scenario
inferencex-agentx-mvp`) against one vLLM server. inference-sim already replays that
benchmark exactly (`blis replay --agentic-scenario inferencex-agentx-mvp`, inference-sim
#1946). What the row does not carry is the input that replay needs: which corpus the
benchmark read, with which seed, and how the server's speculative decoding was set. Those
are in two logs, and this module reads them:

  the CI job log      (GitHub Actions job behind the row's run_url)
      the `aiperf profile` command line   -> corpus, seed, concurrency, duration, entries
      the installed `aiperf==X` line      -> the harness version the replay must match
  the server log      (InferenceX API `/api/v1/server-log?id=<row id>`)
      the vLLM startup banner             -> the vLLM version actually running
      speculative_config in the args line -> method, K, synthetic acceptance length

Why read the acceptance length from the server log rather than choose one. InferenceX
runs speculative arms with `rejection_sample_method: 'synthetic'` and a fixed
`synthetic_acceptance_length`: the acceptance is a setting of the measured run, not an
outcome to fit. Passing that exact value as `--speculative-acceptance-length` replays the
deployment as launched. Choosing any other value would be tuning (the accuracy
dashboard's no-tuning rule), so a speculative run whose acceptance is NOT synthetic is
reported as such and is not given a value here.

Why the vLLM version comes from the banner rather than the image tag. New models first
ship on nightly or model-specific images (`nightly-dev-…-5894fdf`, `kimi-k3`) whose tag
carries no version. The server prints the version it is (`version 0.17.2rc1.dev4996+…`),
so that is the version a version floor should compare.

Nothing here is fetched: callers pass the log text. A field that is absent is absent from
the result rather than defaulted, so a caller can refuse a row it cannot replay exactly.
"""

import re
import shlex

# The AgentX scenario BLIS replays. A command line naming any other scenario is not a
# benchmark `blis replay --agentic-scenario` reproduces.
AGENTX_SCENARIO = "inferencex-agentx-mvp"

# aiperf's alias for a HuggingFace dataset -> the dataset repo it loads. Only date-pinned
# aliases appear here: the rolling alias (…_with_subagents) advances when a new corpus
# drops, so a log naming it does not say which corpus was read.
PINNED_CORPORA = {
    "semianalysis_cc_traces_weka_062126": "semianalysisai/cc-traces-weka-062126",
}


def aiperf_command(job_log: str) -> list[str] | None:
    """The argv of the `aiperf profile` invocation in a CI job log, or None.

    The shell trace prints the command once per invocation, prefixed by the workflow's
    timestamp and `+` (and sometimes a wrapper such as `run_server_client`). The argv
    starts at `profile` so callers read flags by name rather than by position.
    """
    for line in job_log.splitlines():
        i = line.find("aiperf profile ")
        if i < 0 or "--scenario" not in line:
            continue
        try:
            argv = shlex.split(line[i + len("aiperf "):])
        except ValueError:
            continue
        if argv and argv[0] == "profile":
            return argv
    return None


def _flag(argv: list[str], name: str) -> str | None:
    """The value of `--name v` or `--name=v`; the last occurrence wins, as argparse does."""
    val = None
    for k, a in enumerate(argv):
        if a == name and k + 1 < len(argv):
            val = argv[k + 1]
        elif a.startswith(name + "="):
            val = a.split("=", 1)[1]
    return val


def parse_job_log(job_log: str) -> dict:
    """The AgentX inputs recorded in a CI job log.

    Keys present only when the log states them: scenario, corpus_alias, corpus_dataset,
    seed, concurrency, benchmark_duration_s, num_dataset_entries, max_context_length,
    aiperf_version. `corpus_pinned` is False when the corpus alias is not a known
    date-pinned one, which a caller must treat as "corpus unknown".
    """
    out: dict = {}
    m = re.search(r"aiperf==([0-9][0-9A-Za-z.+-]*)", job_log)
    if m:
        out["aiperf_version"] = m.group(1)
    argv = aiperf_command(job_log)
    if argv is None:
        return out
    if (s := _flag(argv, "--scenario")) is not None:
        out["scenario"] = s
    if (alias := _flag(argv, "--public-dataset")) is not None:
        out["corpus_alias"] = alias
        out["corpus_pinned"] = alias in PINNED_CORPORA
        if alias in PINNED_CORPORA:
            out["corpus_dataset"] = PINNED_CORPORA[alias]
    for flag, key in (("--random-seed", "seed"), ("--concurrency", "concurrency"),
                      ("--benchmark-duration", "benchmark_duration_s"),
                      ("--num-dataset-entries", "num_dataset_entries"),
                      ("--max-context-length", "max_context_length")):
        v = _flag(argv, flag)
        if v is not None and re.fullmatch(r"\d+", v):
            out[key] = int(v)
    return out


def _nested(args_line: str, key: str) -> str | None:
    """The text of `'key': {...}` in an args line, matched by brace balance."""
    i = args_line.find(f"'{key}': {{")
    if i < 0:
        return None
    start = args_line.index("{", i)
    depth = 0
    for k in range(start, len(args_line)):
        if args_line[k] == "{":
            depth += 1
        elif args_line[k] == "}":
            depth -= 1
            if depth == 0:
                return args_line[start + 1:k]
    return None


def parse_server_log(server_log: str) -> dict:
    """The vLLM version and speculative setting recorded in a server log.

    Keys present only when the log states them: vllm_version, speculative_method,
    num_speculative_tokens, rejection_sample_method, synthetic_acceptance_length.
    """
    out: dict = {}
    m = re.search(r"\bversion (\d+\.\d+\.\d+[0-9A-Za-z.+-]*)", server_log)
    if m:
        out["vllm_version"] = m.group(1)
    i = server_log.find("non-default args: {")
    spec = _nested(server_log[i:], "speculative_config") if i >= 0 else None
    if spec is not None:
        if (mm := re.search(r"'method':\s*'([A-Za-z_0-9]+)'", spec)):
            out["speculative_method"] = mm.group(1)
        if (mm := re.search(r"'num_speculative_tokens':\s*(\d+)", spec)):
            out["num_speculative_tokens"] = int(mm.group(1))
        if (mm := re.search(r"'rejection_sample_method':\s*'([A-Za-z_0-9]+)'", spec)):
            out["rejection_sample_method"] = mm.group(1)
        if (mm := re.search(r"'synthetic_acceptance_length':\s*([0-9.]+)", spec)):
            out["synthetic_acceptance_length"] = float(mm.group(1))
    return out


def replay_args(job: dict, server: dict) -> tuple[list[str], list[str]]:
    """The `blis replay` arguments for one row, and every reason it cannot be replayed.

    Returns (args, refusals). A non-empty refusals list means the row is not_scored: the
    args are then incomplete and must not be run. The corpus path and sha256 are left to
    the caller (`--corpus`, `--corpus-sha256`), which resolves the pinned dataset to a
    local file.
    """
    refusals = []
    if job.get("scenario") != AGENTX_SCENARIO:
        refusals.append(f"scenario {job.get('scenario')!r} is not {AGENTX_SCENARIO}")
    if not job.get("corpus_pinned"):
        refusals.append(f"corpus {job.get('corpus_alias')!r} is not a date-pinned corpus")
    for key in ("seed", "concurrency"):
        if key not in job:
            refusals.append(f"job log does not state {key}")
    args = ["--agentic-scenario", AGENTX_SCENARIO]
    if "seed" in job:
        args += ["--seed", str(job["seed"])]
    if "concurrency" in job:
        args += ["--concurrency", str(job["concurrency"])]
    if "benchmark_duration_s" in job:
        args += ["--benchmark-duration", f"{job['benchmark_duration_s']}s"]
    if "max_context_length" in job:
        args += ["--max-context-length", str(job["max_context_length"])]
    if "speculative_method" in server:
        if server.get("rejection_sample_method") != "synthetic":
            refusals.append("speculative run without a synthetic acceptance setting: "
                            "its acceptance is an outcome, not a stated value")
        elif "synthetic_acceptance_length" not in server:
            refusals.append("synthetic speculative run with no synthetic_acceptance_length")
        else:
            args += ["--speculative-acceptance-length",
                     repr(server["synthetic_acceptance_length"])]
    return args, refusals
