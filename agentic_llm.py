import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create temp_code directory
CODE_DIR = "temp_code"
os.makedirs(CODE_DIR, exist_ok=True)

def clean_code_output(raw_code: str) -> str:
    """Remove GPT returned ``` blocks, keep only pure code"""
    if "```" in raw_code:
        parts = raw_code.split("```")
        for part in parts:
            if part.strip().startswith("python") or not part.strip().startswith(("```", "")):
                raw_code = part
                break
        raw_code = raw_code.lstrip("python\n").strip()
    return raw_code

def generate_code(prompt: str) -> str:
    """Ask GPT to generate Python code"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    code = response.choices[0].message.content
    return clean_code_output(code)

def save_code(code: str, filename: str = None) -> str:
    """Save code as .py file and return the filename"""
    if filename is None:
        # Find the next available file number
        existing = [f for f in os.listdir(CODE_DIR) if f.startswith("generated_") and f.endswith(".py")]
        next_id = len(existing) + 1
        filename = f"generated_{next_id}.py"

    file_path = os.path.join(CODE_DIR, filename)
    with open(file_path, "w") as f:
        f.write(code)
    return file_path

def run_code(file_path: str) -> str:
    """Run a specific .py file"""
    try:
        result = subprocess.run(
            ["python", file_path],
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"

def auto_fix_code(prompt: str, max_attempts: int = 3):
    """Automatic error fixing loop"""
    code = generate_code(prompt)
    print("=== GPT generated code ===\n", code, "\n")

    file_path = save_code(code)
    print(f"Code saved to: {file_path}\n")

    for attempt in range(1, max_attempts + 1):
        print(f"=== Attempt {attempt} ===")
        output = run_code(file_path)
        print(output)

        if not output.startswith("ERROR"):
            print("✅ Successfully executed!")
            break
        else:
            # Send error message back to GPT for fixing
            fix_prompt = (
                f"The following code has an error:\n\n{code}\n\n"
                f"Error message:\n{output}\n\n"
                "Please fix the code and return ONLY the corrected Python code."
            )
            code = generate_code(fix_prompt)
            file_path = save_code(code)  # Save as new file
            print(f"\n=== GPT fixed code saved to: {file_path} ===\n")

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    auto_fix_code("Write a Python function that loads the iris dataset and chooses the best model hyperparameters to classify iris species.")
