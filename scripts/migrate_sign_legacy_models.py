#!/usr/bin/env python3
"""
Migration: Batch-sign all legacy unsigned .pkl model artifacts.

Run once after deploying the new model_security module to bring all
existing models into the signed state. Safe to run multiple times
(idempotent - skips already-signed models).

Usage:
    python scripts/migrate_sign_legacy_models.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("asmeranda.migrate")


def main() -> None:
    from backend.core.config import settings
    from backend.core.model_security import sign_model_file, verify_model_integrity

    model_dir = Path(settings.data_dir) / "models"
    if not model_dir.exists():
        logger.info("No model directory found at %s — nothing to migrate.", model_dir)
        return

    pkl_files = sorted(model_dir.glob("*.pkl"))
    if not pkl_files:
        logger.info("No .pkl files found in %s", model_dir)
        return

    logger.info("Found %d .pkl files — starting signature migration ...", len(pkl_files))

    signed = 0
    skipped = 0
    failed = 0

    for pkl_path in pkl_files:
        sig_path = pkl_path.with_suffix(".pkl.sig")
        if sig_path.exists():
            # Validate existing sig first
            ok, msg = verify_model_integrity(pkl_path, allow_legacy_unsigned=False)
            if ok:
                logger.debug("SKIP  %s (already signed and valid)", pkl_path.name)
                skipped += 1
                continue
            else:
                logger.warning("Re-signing %s — existing sig invalid: %s", pkl_path.name, msg)

        try:
            sign_model_file(pkl_path, metadata={"migrated": True, "source": "batch_migration"})
            logger.info("SIGNED  %s", pkl_path.name)
            signed += 1
        except Exception as exc:
            logger.error("FAILED  %s: %s", pkl_path.name, exc)
            failed += 1

    logger.info(
        "Migration complete — Signed: %d | Skipped: %d | Failed: %d",
        signed, skipped, failed,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
