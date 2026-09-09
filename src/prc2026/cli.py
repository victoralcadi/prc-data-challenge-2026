"""Command line entry point: `prc2026 <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, data, io_s3, manifest, model


def _months(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v) for v in value.replace(" ", "").split(",") if v]


def cmd_download(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    if args.list:
        print("buckets:", ", ".join(io_s3.list_buckets()) or "(none visible)")
        for name, size in io_s3.list_objects(args.bucket, args.prefix):
            print(f"  {size / 1e6:8.1f} MB  {name}")
        return
    files = io_s3.download(args.bucket, args.prefix, overwrite=args.overwrite)
    print(f"{len(files)} file(s) in {config.RAW_DIR}")
    cmd_status(argparse.Namespace(remote=False, bucket=args.bucket, prefix=args.prefix))


def cmd_status(args: argparse.Namespace) -> None:
    status = manifest.local_status()
    print(f"\nlocal: {config.RAW_DIR}")
    print(status.to_string(index=False))
    print(manifest.describe(status))
    incomplete = status[status["state"].ne("ok")]["object"].tolist()
    if len(incomplete) == len(status):
        print("nothing downloaded yet: run `prc2026 download`")
    elif incomplete:
        print("run `prc2026 download` to fetch:", ", ".join(incomplete))

    if args.remote:
        remote = manifest.remote_status(args.bucket, args.prefix)
        print(f"\nbucket: {args.bucket or config.DATA_BUCKET}")
        print(remote.to_string(index=False))
        unexpected = remote[remote["expected_mb"].isna()]["object"].tolist()
        if unexpected:
            print("not in the documented layout (new or renamed?):", ", ".join(unexpected))


def cmd_audit(args: argparse.Namespace) -> None:
    pd.set_option("display.width", 140, "display.max_rows", 400)
    train_files = sorted(config.RAW_DIR.glob(config.TRAINING_GLOB))[: args.files]
    frames = []
    if train_files:
        frames.append(data.audit(data.load_training(files=train_files), "training"))
    ranking_path = config.RAW_DIR / config.RANKING_FILE
    if ranking_path.exists():
        frames.append(data.audit(data.load_ranking(), "ranking"))
    if not frames:
        raise SystemExit(f"Nothing to audit in {config.RAW_DIR}. Run `prc2026 download` first.")

    report = pd.concat(frames, ignore_index=True)
    wide = report.pivot_table(
        index="column", columns=["dataset", "phase"], values="non_null_pct", sort=False
    )
    print("\nnon-null % per column\n")
    print(wide)

    if ("ranking", "DEP") in wide.columns:
        dep = wide[("ranking", "DEP")]
        blanked = dep[dep.eq(0)].index.tolist()
        print("\nblanked for ranking departures:", blanked or "none")
        for col in ("AOBT_3_flt", "LOBT_flt"):
            if col in dep.index:
                print(f"leak check: {col} populated for {dep[col]:.1f}% of ranking departures")


def cmd_train(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    df = data.load_training(months=_months(args.months))
    period = f"{df[config.MVT_TIME].min()} .. {df[config.MVT_TIME].max()}"
    print(f"loaded {len(df):,} movements, {period}")

    artefact = model.train_validate(df, val_months=tuple(_months(args.val_months) or (1, 7)))
    for key, value in artefact.metrics.items():
        print(f"  {key:>16}: {value:,.2f}")
    print("\ntop features by gain\n")
    print(model.importances(artefact).to_string(index=False))

    if args.refit:
        rounds = int(artefact.best_iteration * args.refit_scale) or model.NUM_ROUNDS
        artefact = model.fit_full(df, rounds)
    path = artefact.save(Path(args.out) if args.out else None)
    print(f"\nsaved {path}")


def cmd_predict(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    artefact = model.Artefact.load(Path(args.model) if args.model else None)
    ranking = data.load_ranking()
    preds = model.predict(artefact, ranking)

    template = data.load_submitting()
    merged = template[[config.ID]].merge(preds, on=config.ID, how="left")
    missing = int(merged[config.TARGET].isna().sum())
    if missing:
        fill = float(preds[config.TARGET].median())
        print(f"warning: {missing} rows had no prediction, filled with median {fill:.0f}s")
        merged[config.TARGET] = merged[config.TARGET].fillna(fill)
    if len(merged) != len(template):
        raise SystemExit("row count changed against the template; submission would be rejected")

    name = args.name or f"{config.TEAM_NAME or 'team'}_v{args.version}.parquet"
    out = config.SUBMISSION_DIR / name
    merged.to_parquet(out, index=False)
    print(
        f"wrote {out} ({len(merged):,} rows)\n"
        f"  predicted taxi-out: mean {merged[config.TARGET].mean():.0f}s "
        f"median {merged[config.TARGET].median():.0f}s"
    )


def cmd_upload(args: argparse.Namespace) -> None:
    name = args.name or f"{config.TEAM_NAME or 'team'}_v{args.version}.parquet"
    path = config.SUBMISSION_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `prc2026 predict --version {args.version}` first.")
    print("uploaded to", io_s3.upload(path, args.bucket))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prc2026", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the challenge data from the OSN bucket")
    p.add_argument("--list", action="store_true", help="only list buckets and objects")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("status", help="check the expected files against disk (and the bucket)")
    p.add_argument("--remote", action="store_true", help="also list the bucket contents")
    p.add_argument("--bucket", default=None)
    p.add_argument("--prefix", default="")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("audit", help="column availability and blanking check")
    p.add_argument("--files", type=int, default=1, help="how many training files to read")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("train", help="validate on held-out months, then optionally refit")
    p.add_argument("--months", default=None, help="training months to load, e.g. 1,2,3")
    p.add_argument("--val-months", default="1,7", help="months held out for validation")
    p.add_argument("--refit", action="store_true", help="refit on all months after validation")
    p.add_argument("--refit-scale", type=float, default=1.1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="score ranking.parquet into a submission file")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("upload", help="push a submission to your team bucket")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--bucket", default=None)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_upload)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
