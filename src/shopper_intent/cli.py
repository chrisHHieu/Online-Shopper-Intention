"""Command line interface: fetch, validate, features, train, evaluate, drift, serve."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from shopper_intent.config import CONFIG_ENV_VAR, Settings, load_config


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="path to a YAML config",
    )

    parser = argparse.ArgumentParser(
        prog="shopper_intent",
        description="Session-level purchase intention pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="path to a YAML config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", parents=[common], help="download the raw dataset")
    fetch_parser.add_argument("--force", action="store_true", help="ignore the cached file")

    subparsers.add_parser("validate", parents=[common], help="check the raw data contract")
    subparsers.add_parser("features", parents=[common], help="build the processed table")

    train_parser = subparsers.add_parser(
        "train", parents=[common], help="train and register models"
    )
    train_parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic", "lightgbm"],
        default=None,
        help="subset of model types to train",
    )
    train_parser.add_argument("--version", default=None, help="registry version to write")

    evaluate_parser = subparsers.add_parser(
        "evaluate", parents=[common], help="score the held-out split and draw figures"
    )
    evaluate_parser.add_argument("--repeats", type=int, default=5, help="permutation repeats")

    sensitivity_parser = subparsers.add_parser(
        "sensitivity",
        parents=[common],
        help="sweep outreach cost on the held-out split",
    )
    sensitivity_parser.add_argument(
        "--costs",
        default=None,
        help="comma separated costs, e.g. 1,5,10,25,50",
    )
    sensitivity_parser.add_argument(
        "--model",
        choices=["logistic", "lightgbm"],
        default=None,
        help="model type to analyse, defaults to serving.model",
    )

    drift_parser = subparsers.add_parser(
        "drift", parents=[common], help="PSI against training data"
    )
    drift_parser.add_argument(
        "--current", type=Path, default=None, help="raw CSV of new sessions to compare"
    )

    serve_parser = subparsers.add_parser("serve", parents=[common], help="run the FastAPI service")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    return parser


def _print_fetch(manifest: dict, cached: bool) -> None:
    state = "cached" if cached else "downloaded"
    print(f"{state} {manifest['row_count']} rows from {manifest['source_url']}")
    print(f"sha256 {manifest['sha256']}")
    print(f"fetched_at {manifest['fetched_at']}")
    if manifest.get("fallback_used"):
        print("note: primary URL failed, fallback URL was used")


def _print_train(results: dict) -> None:
    print(f"version {results['version']}")
    header = f"{'model':<10s} {'threshold':>9s} {'cv_pr_auc':>10s} {'test_pr_auc':>11s} " \
             f"{'test_roc':>9s} {'test_f1':>8s} {'value/1000':>11s}"
    print(header)
    for model_type, metadata in results["models"].items():
        cv = metadata["metrics"]["cv"]
        test = metadata["metrics"]["test"]
        print(
            f"{model_type:<10s} {metadata['threshold']:>9.4f} {cv['pr_auc']:>10.4f} "
            f"{test['pr_auc']:>11.4f} {test['roc_auc']:>9.4f} {test['f1']:>8.4f} "
            f"{test['net_value_per_1000_sessions']:>11.2f}"
        )


def _print_evaluate(report: dict, settings: Settings) -> None:
    print(f"primary model {report['primary_model']}")
    for model_type, payload in report["models"].items():
        metrics = payload["metrics_test"]
        policy = payload["policy_test"]
        print(
            f"{model_type:<10s} threshold {payload['threshold']:.4f} "
            f"precision {metrics['precision']:.4f} recall {metrics['recall']:.4f} "
            f"f1 {metrics['f1']:.4f} pr_auc {metrics['pr_auc']:.4f} "
            f"roc_auc {metrics['roc_auc']:.4f} brier {metrics['brier']:.4f} "
            f"ks {metrics['ks_at_threshold']:.4f}"
        )
        print(
            f"{'':<10s} net value per 1000 sessions {metrics['net_value_per_1000_sessions']:.2f} "
            f"(vs 0.5: {policy['gap_vs_default_per_1000_sessions']:+.2f}, "
            f"vs always contact: {policy['gap_vs_always_contact_per_1000_sessions']:+.2f})"
        )
    print(f"metrics  {settings.reports_dir / 'metrics.json'}")
    print(f"figures  {settings.figures_dir}")
    print(f"decile lift table {settings.reports_dir / 'decile_lift.csv'}")
    print(f"cost sensitivity  {settings.reports_dir / 'cost_sensitivity.csv'}")


def _print_sensitivity(result: dict, settings: Settings) -> None:
    print(
        f"model {result['model_type']} version {result['version']} "
        f"value per conversion {result['value_per_conversion']:.2f}"
    )
    header = (
        f"{'cost':>6s} {'thr':>7s} {'contact':>8s} {'precision':>9s} {'recall':>7s} "
        f"{'model/1000':>12s} {'always/1000':>12s} {'delta':>10s}"
    )
    print(header)
    for row in result["rows"]:
        print(
            f"{row['cost_per_outreach']:>6.0f} {row['threshold']:>7.4f} "
            f"{row['contact_rate']:>8.4f} {row['precision']:>9.4f} {row['recall']:>7.4f} "
            f"{row['net_value_per_1000_model']:>12.2f} "
            f"{row['net_value_per_1000_always']:>12.2f} "
            f"{row['delta_per_1000']:>10.2f}"
        )
    print(f"csv     {settings.reports_dir / 'cost_sensitivity.csv'}")
    print(f"figure  {settings.figures_dir / 'cost_sensitivity.png'}")


def _print_drift(report: dict, settings: Settings) -> None:
    summary = report["summary"]
    print(
        f"current {report['current']} rows {report['current_rows']} "
        f"reference rows {report['reference_rows']}"
    )
    print(
        f"features {summary['features_checked']} significant {summary['significant']} "
        f"moderate {summary['moderate']} max_psi {summary['max_psi']:.4f}"
    )
    for row in report["features"][:5]:
        print(f"{row['status']:<11s} psi {row['psi']:>7.4f}  {row['feature']}")
    print(f"report {settings.reports_dir / 'drift.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(getattr(args, "config", None))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.command == "fetch":
        from shopper_intent.data.fetch import FetchError, cached_manifest, fetch

        was_cached = not args.force and cached_manifest(config) is not None
        try:
            manifest = fetch(config, force=args.force)
        except FetchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_fetch(manifest, cached=was_cached)
        return 0

    if args.command == "validate":
        from shopper_intent.data.validate import validate

        try:
            report = validate(config)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(report.render())
        return 0 if report.ok else 1

    if args.command == "features":
        from shopper_intent.features.build import read_feature_schema, write_processed

        try:
            path = write_processed(config)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        schema = read_feature_schema(config)
        print(f"wrote {path}")
        print(
            f"rows {schema['rows']} train {schema['train_rows']} test {schema['test_rows']} "
            f"features {len(schema['features'])} positive_rate {schema['positive_rate']:.4f}"
        )
        return 0

    if args.command == "train":
        from shopper_intent.models.train import train

        try:
            results = train(config, model_types=args.models, version=args.version)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_train(results)
        for model_type, path in results["artifacts"].items():
            print(f"{model_type} artifact {path}")
        return 0

    if args.command == "evaluate":
        from shopper_intent.models.evaluate import evaluate

        try:
            report = evaluate(config, n_repeats=args.repeats)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_evaluate(report, config)
        return 0

    if args.command == "sensitivity":
        from shopper_intent.models.sensitivity import cost_sensitivity

        costs = None
        if args.costs:
            try:
                costs = [float(value) for value in args.costs.split(",")]
            except ValueError:
                print("error: --costs must be comma separated numbers", file=sys.stderr)
                return 1
        try:
            result = cost_sensitivity(config, costs=costs, model_type=args.model)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_sensitivity(result, config)
        return 0

    if args.command == "drift":
        from shopper_intent.monitor.drift import drift

        try:
            report = drift(config, current_path=args.current)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_drift(report, config)
        return 0

    if args.command == "serve":
        import uvicorn

        os.environ[CONFIG_ENV_VAR] = str(config.source_path)
        host = args.host or config.serving.host
        port = args.port or config.serving.port
        print(f"serving {config.serving.model} on http://{host}:{port}")
        uvicorn.run("shopper_intent.serving.app:app", host=host, port=port, log_level="info")
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2
