import json

def convert(py_path, java_path, out_path):
    with open(py_path)   as f_py, \
         open(java_path) as f_java, \
         open(out_path, "w") as f_out:

        py_lines   = f_py.read().strip().split("\n")
        java_lines = f_java.read().strip().split("\n")

        assert len(py_lines) == len(java_lines), \
            f"Mismatch: {len(py_lines)} python vs {len(java_lines)} java lines"

        for py, java in zip(py_lines, java_lines):
            obj = {"python": py.strip(), "java": java.strip()}
            f_out.write(json.dumps(obj) + "\n")

    print(f"Written {len(py_lines)} pairs → {out_path}")

# adjust paths to wherever your Java-Python folder is
BASE = "data/"

convert(BASE + "train-Java-Python-tok.py",
        BASE + "train-Java-Python-tok.java",
        "train.jsonl")

convert(BASE + "val-Java-Python-tok.py",
        BASE + "val-Java-Python-tok.java",
        "val.jsonl")

convert(BASE + "test-Java-Python-tok.py",
        BASE + "test-Java-Python-tok.java",
        "test.jsonl")