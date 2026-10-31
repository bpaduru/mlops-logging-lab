import argparse
import json
import logging
import time
from datetime import datetime

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from logger_setup import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # every run gets its own id and its own log file
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger("train", log_file=f"logs/run_{run_id}.log")

    # send python warnings (like sklearn convergence warnings) into the same log files
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.propagate = False
    for handler in logger.handlers:
        warnings_logger.addHandler(handler)

    logger.info("starting run %s", run_id)
    logger.info(
        "params: C=%s, max_iter=%s, test_size=%s",
        args.C, args.max_iter, args.test_size,
    )

    try:
        X, y = load_wine(return_X_y=True)
        logger.debug("loaded wine data: %d rows, %d features", X.shape[0], X.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        logger.info("train rows: %d, test rows: %d", len(X_train), len(X_test))

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=args.C, max_iter=args.max_iter),
        )

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        logger.info("training finished in %.3f seconds", train_time)

        acc = accuracy_score(y_test, model.predict(X_test))
        logger.info("test accuracy: %.4f", acc)

        if acc < 0.9:
            logger.warning("accuracy is below 0.9, check the params")

    except Exception:
        logger.exception("run %s failed", run_id)
        raise SystemExit(1)

    summary = {
        "run_id": run_id,
        "C": args.C,
        "max_iter": args.max_iter,
        "test_size": args.test_size,
        "accuracy": round(acc, 4),
        "train_time_sec": round(train_time, 3),
    }
    with open("logs/runs.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")

    logger.info("saved run summary to logs/runs.jsonl")


if __name__ == "__main__":
    main()
