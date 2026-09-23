"""Audit warm-start/trained actor drift on deterministic CPU probe states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evotac.learning.actor_drift import audit_actor_drift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    report = audit_actor_drift(args.trained_checkpoint, args.warmstart_checkpoint,
                               candidate_checkpoint=args.candidate_checkpoint,
                               count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
