"""Access to the OpenSky Network S3 buckets holding the challenge data.

Mirrors what `mc` does, so you do not need the MinIO CLI installed:

    mc alias set dc26 https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY
    mc cp --recursive dc26/competition-data/ data/raw/
"""

from __future__ import annotations

from pathlib import Path

from . import config


def client():
    """Return a MinIO client for the challenge endpoint."""
    from minio import Minio

    if not (config.ACCESS_KEY and config.SECRET_KEY):
        raise RuntimeError(
            "PRC_ACCESS_KEY / PRC_SECRET_KEY are not set. Copy .env.example to .env "
            "and paste the access keys generated on your OSN account."
        )
    return Minio(
        config.S3_ENDPOINT,
        access_key=config.ACCESS_KEY,
        secret_key=config.SECRET_KEY,
        secure=True,
    )


def list_buckets() -> list[str]:
    return [b.name for b in client().list_buckets()]


def list_objects(bucket: str | None = None, prefix: str = "") -> list[tuple[str, int]]:
    bucket = bucket or config.DATA_BUCKET
    objects = client().list_objects(bucket, prefix=prefix, recursive=True)
    return [(o.object_name, o.size or 0) for o in objects]


def download(bucket: str | None = None, prefix: str = "", dest: Path | None = None,
             overwrite: bool = False) -> list[Path]:
    """Download every object under `prefix` into `dest` (default `data/raw`)."""
    bucket = bucket or config.DATA_BUCKET
    dest = dest or config.RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)
    mc = client()
    written: list[Path] = []
    for name, size in list_objects(bucket, prefix):
        target = dest / Path(name).name
        if target.exists() and not overwrite and target.stat().st_size == size:
            print(f"  skip {target.name} (already present)")
            written.append(target)
            continue
        print(f"  get  {name} -> {target} ({size / 1e6:.1f} MB)")
        mc.fget_object(bucket, name, str(target))
        written.append(target)
    return written


def upload(path: Path, bucket: str | None = None, object_name: str | None = None) -> str:
    """Upload a submission file. Returns the object name written."""
    bucket = bucket or config.SUBMISSION_BUCKET
    if not bucket:
        raise RuntimeError(
            "PRC_SUBMISSION_BUCKET is not set (each team submits to its own bucket)."
        )
    object_name = object_name or path.name
    client().fput_object(bucket, object_name, str(path))
    return f"{bucket}/{object_name}"
