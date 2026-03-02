# preprocess_input.py
import sys

def preprocess(template_path, output_path, variables):
    with open(template_path, "r") as f:
        content = f.read()

    for key, value in variables.items():
        content = content.replace(f"${{{key}}}", value)

    with open(output_path, "w") as f:
        f.write(content)

# Usage:
# python preprocess_input.py input_template output_input {"OUTPUT_BASE": "/your/path"}
if __name__ == "__main__":
    import json
    preprocess(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))

