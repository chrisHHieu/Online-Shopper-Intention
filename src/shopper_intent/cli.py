"""Command line interface: fetch, validate, features, train, evaluate, drift, serve."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from shopper_intent.config import CONFIG_ENV_VAR, MODEL_NAMES, Settings, load_config


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
        choices=list(MODEL_NAMES),
        default=None,
        help="subset of model types to train",
    )
    train_parser.add_argument("--version", default=None, help="registry version to write")
    train_parser.add_argument(
        "--track",
        action="store_true",
        default=None,
        help="log the run to MLflow and register the models",
    )
    train_parser.add_argument(
        "--no-track",
        dest="track",
        action="store_false",
        help="skip MLflow even when tracking.enabled is true in the config",
    )

    tune_parser = subparsers.add_parser(
        "tune",
        parents=[common],
        help="search hyperparameters with Optuna and write a tuned config",
    )
    tune_parser.add_argument("--trials", type=int, default=None, help="trials per model")
    tune_parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_NAMES),
        default=None,
        help="models to tune, defaults to tuning.models or serving.model",
    )
    tune_parser.add_argument("--timeout", type=int, default=None, help="seconds per study")
    tune_parser.add_argument(
        "--out", type=Path, default=None, help="tuned config path, defaults to configs/tuned.yaml"
    )

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
    drift_parser.add_argument(
        "--evidently",
        action="store_true",
        help="also write the Evidently HTML drift report",
    )
    drift_parser.add_argument(
        "--reference-months", default=None, help="comma separated months for the reference period"
    )
    drift_parser.add_argument(
        "--current-months", default=None, help="comma separated months for the current period"
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


def _months(value: str | None) -> list[str] | None:
    if not value:
        return None
    months = [part.strip() for part in value.split(",") if part.strip()]
    return months or None


def _print_train(results: dict) -> None:
    print(f"version {results['version']}")
    header = (
        f"{'model':<10s} {'calib':<9s} {'threshold':>9s} {'cv_pr_auc':>10s} "
        f"{'test_pr_auc':>11s} {'test_roc':>9s} {'test_brier':>10s} {'test_ece':>8s} "
        f"{'value/1000':>11s} {'coverage':>9s}"
    )
    print(header)
    for model_type, metadata in results["models"].items():
        cv = metadata["metrics"]["cv"]
        test = metadata["metrics"]["test"]
        conformal = metadata["metrics"].get("conformal_test") or {}
        coverage = conformal.get("coverage")
        print(
            f"{model_type:<10s} {str(metadata['calibration']['method']):<9s} "
            f"{metadata['threshold']:>9.4f} {cv['pr_auc']:>10.4f} "
            f"{test['pr_auc']:>11.4f} {test['roc_auc']:>9.4f} {test['brier']:>10.4f} "
            f"{test['ece']:>8.4f} {test['net_value_per_1000_sessions']:>11.2f} "
            f"{coverage if coverage is not None else float('nan'):>9.4f}"
        )
    print(f"champion {results['champion']} by {results['selection_rule']}")
    tracking = results.get("tracking")
    if tracking:
        if tracking.get("run_id"):
            print(f"mlflow run {tracking['run_id']}")
            for model_type, entry in tracking["models"].items():
                print(
                    f"mlflow {model_type:<10s} {entry.get('registered_model', '-')} "
                    f"version {entry.get('version', '-')} alias {entry.get('alias', '-')}"
                )
        print(f"mlflow {tracking.get('fallback') or 'ok'} ({tracking.get('tracking_uri')})")


def _print_evaluate(report: dict, settings: Settings) -> None:
    print(f"primary model {report['primary_model']}  champion {report.get('champion')}")
    header = (
        f"{'model':<10s} {'calib':<9s} {'thr':>7s} {'roc_auc':>8s} {'pr_auc':>8s} "
        f"{'brier':>8s} {'ece':>8s} {'profit/1000':>12s} {'coverage':>9s}"
    )
    print(header)
    for row in report["comparison"]:
        coverage = row["conformal_coverage"]
        print(
            f"{row['model']:<10s} {str(row['calibration']):<9s} {row['threshold']:>7.4f} "
            f"{row['roc_auc']:>8.4f} {row['pr_auc']:>8.4f} {row['brier']:>8.4f} "
            f"{row['ece']:>8.4f} {row['net_value_per_1000_sessions']:>12.2f} "
            f"{coverage if coverage is not None else float('nan'):>9.4f}"
        )
    print("held-out split, threshold chosen on out-of-fold training probabilities")
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


def _print_tune(results: dict, settings: Settings) -> None:
    print(
        f"trials requested {results['trials_requested']} per model, "
        f"timeout {results['timeout_seconds']}s, {results['duration_seconds']}s total"
    )
    header = f"{'model':<10s} {'done':>5s} {'pruned':>7s} {'pr_auc_oof':>11s} {'value/1000':>11s}"
    print(header)
    for model_type, payload in results["models"].items():
        pr_auc = payload["best_pr_auc_oof"]
        print(
            f"{model_type:<10s} {payload['completed']:>5d} {payload['pruned']:>7d} "
            f"{pr_auc if pr_auc is not None else float('nan'):>11.4f} "
            f"{payload['best_value_per_1000']:>11.2f}"
        )
        for key, value in sorted(payload["best_params"].items()):
            print(f"    {key} = {value}")
    print(f"tuned config {results['path']}")
    print(f"studies      {settings.optuna_storage}")


def _print_evidently(payload: dict) -> None:
    features = payload["features"]
    print(
        f"evidently {payload['evidently_version']} reference {payload['reference']} "
        f"({payload['rows']['reference']} rows) current {payload['current']} "
        f"({payload['rows']['current']} rows)"
    )
    print(
        f"features {features['columns_checked']} drifted "
        f"{features['drifted_count']} share {features['drifted_share']:.4f}"
    )
    for row in features["details"][:8]:
        state = "drift" if row["drifted"] else "stable"
        print(f"{state:<7s} p {row['value']:>8.5f}  {row['column']}")
    target = payload["target"]
    delta = target["delta"]
    print(
        f"target {target['column']} rate {target['reference_rate']:.4f} -> "
        f"{target['current_rate']:.4f} delta {delta:+.4f} psi {target['psi']:.4f}"
    )
    print(f"html {payload['html']}")
    print(f"json {payload['json']}")


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
        ok = report.ok
        if config.processed_path.is_file():
            from shopper_intent.data.validate import validate_processed

            processed = validate_processed(config)
            print()
            print("processed table")
            print(processed.render())
            ok = ok and processed.ok
        return 0 if ok else 1

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
            results = train(
                config, model_types=args.models, version=args.version, track=args.track
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_train(results)
        for model_type, path in results["artifacts"].items():
            print(f"{model_type} artifact {path}")
        return 0

    if args.command == "tune":
        from shopper_intent.tuning.search import tune

        try:
            results = tune(
                config,
                models=args.models,
                trials=args.trials,
                timeout=args.timeout,
                out=args.out,
                command="tune " + " ".join(argv if argv is not None else sys.argv[1:]),
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_tune(results, config)
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
        if args.evidently:
            from shopper_intent.monitor.evidently_report import (
                EvidentlyUnavailable,
                drift_evidently,
            )

            try:
                payload = drift_evidently(
                    config,
                    current_path=args.current,
                    reference_months=_months(args.reference_months),
                    current_months=_months(args.current_months),
                )
            except EvidentlyUnavailable as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print()
            _print_evidently(payload)
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
