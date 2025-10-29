#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

def load_pipeline(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)

def prepare_input(record: dict):
    # Return a list-of-dicts by default (works with DictVectorizer / sklearn pipelines
    # that expect iterable of mappings). If callers need a DataFrame they can convert
    # it themselves or the score() function will try a fallback.
    if isinstance(record, dict):
        return [record]
    # If user passed a JSON array already
    if isinstance(record, list):
        return record
    # Otherwise wrap whatever was provided
    return [record]

def score(pipeline: Any, X):
    def _call(pX):
        if hasattr(pipeline, "predict_proba"):
            out = pipeline.predict_proba(pX)
            try:
                # binary class -> return prob of positive class
                if hasattr(out, "ndim") and out.ndim == 2 and out.shape[1] == 2:
                    return out[:, 1].tolist()
            except Exception:
                pass
            try:
                return out.tolist()
            except Exception:
                return list(out)
        if hasattr(pipeline, "predict"):
            out = pipeline.predict(pX)
            try:
                return out.tolist()
            except Exception:
                return list(out)
        raise RuntimeError("Loaded object has neither predict_proba nor predict")

    try:
        return _call(X)
    except Exception as e:
        # Common sklearn DictVectorizer error arises when a DataFrame (iterable of column names)
        # is passed where an iterable of dicts is expected. Try converting DataFrame -> records.
        msg = str(e)
        try:
            import pandas as pd  # type: ignore
            if isinstance(X, pd.DataFrame):
                return _call(X.to_dict(orient="records"))
        except Exception:
            pass
        # As last resort, if X is list-of-mappings expected to be a DataFrame, try to convert
        try:
            import pandas as pd  # type: ignore
            if isinstance(X, list):
                return _call(pd.DataFrame(X))
        except Exception:
            pass
        # Re-raise original exception if fallbacks failed
        raise

def main():
    parser = argparse.ArgumentParser(description="Score a single record with a pickled pipeline.")
    parser.add_argument(
        "record",
        nargs="?",
        help='JSON string with the record, e.g. \'{"lead_source":"paid_ads","number_of_courses_viewed":2,"annual_income":79276.0}\'',
    )
    parser.add_argument("--pipeline", "-p", default="pipeline_v1.bin", help="Path to pipeline file (default: pipeline_v1.bin)")
    args = parser.parse_args()

    if not args.record:
        print("Provide a JSON record as the first argument.", file=sys.stderr)
        parser.print_help()
        sys.exit(2)

    try:
        record = json.loads(args.record)
        if not isinstance(record, dict):
            raise ValueError("Record must be a JSON object")
    except Exception as e:
        print(f"Invalid JSON record: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        pipeline = load_pipeline(Path(args.pipeline))
    except Exception as e:
        print(f"Error loading pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    X = prepare_input(record)
    try:
        result = score(pipeline, X)
    except Exception as e:
        print(f"Error scoring record: {e}", file=sys.stderr)
        sys.exit(1)

    # Print compact JSON output
    out = {"record": record, "score": result}
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()