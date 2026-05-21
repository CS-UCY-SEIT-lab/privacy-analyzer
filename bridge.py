import json
import sys
import traceback
import pandas as pd

from api import (
    build_stage1_result,
    process_requirement_for_ui,
    get_stage1_results,
    run_stage2_analysis,
    extract_texts_from_df,
)

def error(msg, trace=None):
    out = {"error": msg}
    if trace:
        out["trace"] = trace
    print(json.dumps(out))
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        error("Missing endpoint")

    endpoint = sys.argv[1]

    try:
        if endpoint == "health":
            print(json.dumps({
                "status": "ok",
                "device": "cpu",
                "stage1_loaded": True,
                "stage2_loaded": True
            }))
            return

        if endpoint == "predict":
            raw = sys.stdin.read().strip()
            if not raw:
                error("Missing JSON body")

            data = json.loads(raw)

            texts = []
            override_label = data.get("override_label", None)
            proceed_to_stage2 = bool(data.get("proceed_to_stage2", False))
            decompose = bool(data.get("decompose", True))

            if "text" in data:
                text = str(data["text"]).strip()
                if text:
                    texts = [text]

            elif "texts" in data:
                if not isinstance(data["texts"], list):
                    error("'texts' must be a list")
                texts = [str(t).strip() for t in data["texts"] if str(t).strip()]

            if not texts:
                error("No valid input text provided")

            if len(texts) == 1:
                result = process_requirement_for_ui(
                    texts[0],
                    override_label=override_label,
                    proceed_to_stage2=proceed_to_stage2,
                    decompose=decompose
                )
                print(json.dumps(result))
                return

            results = get_stage1_results(texts, decompose=False)
            print(json.dumps({"results": results}))
            return

        if endpoint == "analyze-stage2":
            raw = sys.stdin.read().strip()
            if not raw:
                error("Missing JSON body")

            data = json.loads(raw)

            text = str(data.get("text", "")).strip()
            if not text:
                error("No valid input text provided")

            override_label = data.get("override_label", None)
            decompose = bool(data.get("decompose", True))
            force_stage2 = bool(data.get("force_stage2", False))

            stage1 = build_stage1_result(text, override_label=override_label)

            if stage1["final_decision"]["label"] == 0 and not force_stage2:
                print(json.dumps({
                    "input_text": text,
                    "stage1": stage1,
                    "stage2": {
                        "status": "skipped",
                        "reason": "This requirement is currently marked as non-privacy, so Stage 2 was not run.",
                        "matches": [],
                        "match_count": 0,
                        "top_scores": [],
                        "evidence": {},
                        "decomposed": False,
                        "sub_requirements": []
                    }
                }))
                return

            stage2 = run_stage2_analysis(text, decompose=decompose)

            print(json.dumps({
                "input_text": text,
                "stage1": stage1,
                "stage2": stage2
            }))
            return

        if endpoint == "predict-file":
            if len(sys.argv) < 3:
                error("Missing file path")

            file_path = sys.argv[2]
            filename = file_path.lower()

            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                texts = [line.strip() for line in content.splitlines() if line.strip()]

            elif filename.endswith(".csv"):
                df = pd.read_csv(file_path)
                texts = extract_texts_from_df(df)

            elif filename.endswith(".xlsx"):
                df = pd.read_excel(file_path)
                texts = extract_texts_from_df(df)

            else:
                error("Unsupported file type. Use .txt, .csv, or .xlsx")

            if not texts:
                error("No valid requirements found in file")

            results = get_stage1_results(texts, decompose=False)

            print(json.dumps({
                "results": results,
                "message": "Stage 1 completed. Review the classification of each requirement before proceeding to Stage 2."
            }))
            return

        error("Invalid endpoint")

    except Exception as e:
        error(str(e), traceback.format_exc())

if __name__ == "__main__":
    main()