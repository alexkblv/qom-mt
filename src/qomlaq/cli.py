"""Command line entry point. Every command takes an experiment from experiments/.

    qomlaq experiments                        # list experiments
    qomlaq runs EXPERIMENT                    # runs, and which are trained and scored
    qomlaq build [EXPERIMENT ...]             # corpus + splits, checked against pins
    qomlaq splits EXPERIMENT                  # split sizes, hashes and pin status
    qomlaq train EXPERIMENT RUN_ID ...        # needs the [train] extra and a GPU
    qomlaq evaluate EXPERIMENT RUN_ID ...     # score runs on their own test sets
    qomlaq evaluate EXPERIMENT --evaluation NAME
    qomlaq report EXPERIMENT                  # tables in artifacts/tables/EXPERIMENT/
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_experiments(args) -> int:
    from .experiments import list_experiments, load_experiment

    names = list_experiments()
    if not names:
        print("no experiments; add a .toml file to experiments/")
        return 1
    for name in names:
        exp = load_experiment(name)
        print(f"{name:20s} {exp.corpus:10s} {exp.model:12s} {len(exp.runs):3d} runs  "
              f"{exp.description}")
    return 0


def _cmd_runs(args) -> int:
    from .pipeline import run_status

    print(run_status(args.experiment, artifacts_dir=args.artifacts_dir).to_string(index=False))
    return 0


def _cmd_build(args) -> int:
    from .experiments import list_experiments
    from .pipeline import build

    for name in args.experiments or list_experiments():
        print(f"=== {name}")
        build(name, data_dir=args.data_dir, artifacts_dir=args.artifacts_dir,
              rebuild=args.rebuild)
        print()
    return 0


def _cmd_splits(args) -> int:
    from .pipeline import split_status

    print(split_status(args.experiment, artifacts_dir=args.artifacts_dir).to_string(index=False))
    return 0


def _selected_runs(args) -> list[str]:
    from .experiments import load_experiment

    if args.all:
        return list(load_experiment(args.experiment).runs_by_id)
    return list(args.run_ids)


def _cmd_train(args) -> int:
    from .pipeline import train

    run_ids = _selected_runs(args)
    if not run_ids:
        print("name the run(s) to train, or pass --all", file=sys.stderr)
        return 2
    for run_id in run_ids:
        train(args.experiment, run_id, data_dir=args.data_dir,
              artifacts_dir=args.artifacts_dir, resume=args.resume)
    return 0


def _cmd_evaluate(args) -> int:
    from .pipeline import evaluate_run, run_evaluation

    run_ids = _selected_runs(args)
    if not run_ids and not args.evaluation:
        print("name the run(s) to evaluate, pass --all, or pass --evaluation NAME",
              file=sys.stderr)
        return 2
    for run_id in run_ids:
        evaluate_run(args.experiment, run_id, partition=args.partition,
                     data_dir=args.data_dir, artifacts_dir=args.artifacts_dir)
    if args.evaluation:
        run_evaluation(args.experiment, args.evaluation, data_dir=args.data_dir,
                       artifacts_dir=args.artifacts_dir)
    return 0


def _cmd_report(args) -> int:
    from .pipeline import report

    for name, path in report(args.experiment, artifacts_dir=args.artifacts_dir).items():
        print(f"{name:28s} {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Progress is the point of a long-running command; without this a piped run shows
    # nothing until it finishes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(prog="qomlaq", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"qomlaq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # Accepted by every command, after its name: `qomlaq build --data-dir DIR`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=None, help="raw corpus sources")
    common.add_argument("--artifacts-dir", default=None, help="where artifacts are written")

    def command(name: str, **kwargs):
        return sub.add_parser(name, parents=[common], **kwargs)

    command("experiments", help="list experiments").set_defaults(func=_cmd_experiments)

    runs = command("runs", help="list an experiment's runs and their status")
    runs.add_argument("experiment")
    runs.set_defaults(func=_cmd_runs)

    build = command("build", help="build corpus and splits")
    build.add_argument("experiments", nargs="*", help="default: every experiment")
    build.add_argument("--rebuild", action="store_true", help="ignore any cached corpus")
    build.set_defaults(func=_cmd_build)

    splits = command("splits", help="split sizes, hashes and pin status")
    splits.add_argument("experiment")
    splits.set_defaults(func=_cmd_splits)

    train = command("train", help="train runs (needs a GPU)")
    train.add_argument("experiment")
    train.add_argument("run_ids", nargs="*")
    train.add_argument("--all", action="store_true", help="every run of the experiment")
    train.add_argument("--resume", action="store_true", help="resume from the last checkpoint")
    train.set_defaults(func=_cmd_train)

    evaluate = command("evaluate", help="score trained runs")
    evaluate.add_argument("experiment")
    evaluate.add_argument("run_ids", nargs="*")
    evaluate.add_argument("--all", action="store_true", help="every run of the experiment")
    evaluate.add_argument("--partition", default="test", choices=("dev", "test"))
    evaluate.add_argument("--evaluation", default=None,
                          help="one of the experiment's named evaluations")
    evaluate.set_defaults(func=_cmd_evaluate)

    report = command("report", help="write an experiment's tables")
    report.add_argument("experiment")
    report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # surfaced as a message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
