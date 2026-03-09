# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import argparse
import json


def main():
    p = argparse.ArgumentParser(description="Create a light JSONL with only docids.")
    p.add_argument("--input", required=True, help="Path to full JSONL")
    p.add_argument("--output", required=True, help="Path to write the light JSONL")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                obj = json.loads(line)
                for k in ["gold_docs", "evidence_docs", "negative_docs"]:
                    obj[k] = [d.get("docid") for d in obj.get(k) if d.get("docid") is not None]

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
