#!/usr/bin/env python3
"""Behavioural tests for reading agentic replay arguments from a row's two logs.

The fixtures are lines from real logs (2026-10), shortened to what the parser reads:
Kimi-K3 row 439936 (CI job 95056545680, server log id 439936) and DeepSeek-V4.1-Flash row
442752 (CI job 106118932015). Each check is a way the replay would be wrong while every
field still parsed: the wrong corpus, a seed nobody ran, an acceptance we chose.

Run: python3 scripts/test_extract_agentic_replay_args.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_agentic_replay_args import (  # noqa: E402
    parse_job_log, parse_server_log, replay_args)

KIMI_JOB = (
    "2026-08-15T15:52:01.1Z  + aiperf==0.12.0 (from file:///workspace/utils/aiperf)\n"
    "2026-08-15T15:58:59.2Z + /tmp/inferencex-agentic-43701/venv/bin/aiperf profile "
    "--scenario inferencex-agentx-mvp --url http://localhost:8888 --endpoint "
    "/v1/chat/completions --endpoint-type chat --streaming --model moonshotai/Kimi-K3 "
    "--concurrency 2 --benchmark-duration 3600 --stats-interval 30 --random-seed 42 "
    "--failed-request-threshold 0.10 --num-dataset-entries 393 --slice-duration 1.0 "
    "--output-artifact-dir /workspace/results/aiperf_artifacts "
    "--public-dataset semianalysis_cc_traces_weka_062126\n"
)

# The V4.1 run wraps the command in a helper, which the parser must see through.
V41_JOB = (
    "2026-10-09T01:00:00.0Z + run_server_client /tmp/x/venv/bin/aiperf profile --scenario "
    "inferencex-agentx-mvp --model deepseek-ai/DeepSeek-V4.1-Flash --concurrency 1 "
    "--benchmark-duration 3600 --random-seed 42 --num-dataset-entries 393 "
    "--public-dataset semianalysis_cc_traces_weka_062126\n"
)

KIMI_SERVER = (
    "(APIServer pid=1) INFO [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  "
    "version 0.17.2rc1.dev4996+g5894fdf98\n"
    "(APIServer pid=1) INFO [api_utils.py:273] non-default args: {'model': "
    "'/scratch/models/Kimi-K3', 'tensor_parallel_size': 8, 'max_num_seqs': 4, "
    "'speculative_config': {'method': 'dspark', 'model': '/data/models/Kimi-K3-DSpark', "
    "'num_speculative_tokens': 7, 'draft_sample_method': 'probabilistic', "
    "'rejection_sample_method': 'synthetic', 'synthetic_acceptance_length': 3.84}, "
    "'kv_transfer_config': KVTransferConfig(kv_connector='MooncakeStoreConnector')}\n"
    "INFO [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.78\n"
)

FAILURES = []


def ck(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    job = parse_job_log(KIMI_JOB)
    ck("the corpus is the date-pinned dataset the benchmark read",
       job.get("corpus_dataset") == "semianalysisai/cc-traces-weka-062126" and job.get("corpus_pinned"),
       repr(job))
    ck("seed, concurrency and duration come from the command line",
       (job.get("seed"), job.get("concurrency"), job.get("benchmark_duration_s")) == (42, 2, 3600),
       repr(job))
    ck("the harness version is read", job.get("aiperf_version") == "0.12.0", repr(job.get("aiperf_version")))

    v41 = parse_job_log(V41_JOB)
    ck("a wrapped command (run_server_client …) is still found",
       (v41.get("concurrency"), v41.get("seed")) == (1, 42), repr(v41))

    # The rolling alias names whichever corpus is current, so it identifies none.
    rolling = parse_job_log(KIMI_JOB.replace("semianalysis_cc_traces_weka_062126",
                                             "semianalysis_cc_traces_weka_with_subagents"))
    _, refusals = replay_args(rolling, {})
    ck("a rolling corpus alias is refused, not resolved to the latest corpus",
       rolling.get("corpus_pinned") is False and any("date-pinned" in r for r in refusals),
       repr(refusals))

    server = parse_server_log(KIMI_SERVER)
    ck("the vLLM version is the banner's, not the image tag's",
       server.get("vllm_version") == "0.17.2rc1.dev4996+g5894fdf98", repr(server.get("vllm_version")))
    ck("the acceptance is the launch setting, not the logged outcome",
       server.get("synthetic_acceptance_length") == 3.84, repr(server))
    ck("the draft's own 'model' key does not confuse the speculative fields",
       (server.get("speculative_method"), server.get("num_speculative_tokens")) == ("dspark", 7),
       repr(server))

    args, refusals = replay_args(job, server)
    ck("a pinned, synthetic-acceptance row replays with no refusal", refusals == [], repr(refusals))
    ck("the replay carries the run's seed, lanes and acceptance length",
       args[args.index("--seed") + 1] == "42" and args[args.index("--concurrency") + 1] == "2"
       and args[args.index("--speculative-acceptance-length") + 1] == "3.84", repr(args))

    # A speculative run whose acceptance was not preset has no stated value to pass;
    # supplying one would be fitting the model to the measurement.
    natural = parse_server_log(KIMI_SERVER.replace("'rejection_sample_method': 'synthetic', ", "")
                               .replace(", 'synthetic_acceptance_length': 3.84", ""))
    args, refusals = replay_args(job, natural)
    ck("a speculative run without a preset acceptance is refused, never given a value",
       "--speculative-acceptance-length" not in args and any("synthetic" in r for r in refusals),
       repr(refusals))

    ck("a job log with no aiperf command yields no seed and is refused",
       replay_args(parse_job_log("no command here\n"), {})[1] != [])

    if FAILURES:
        print(f"\n{len(FAILURES)} failed: {FAILURES}")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
