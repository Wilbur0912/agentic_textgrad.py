import os
import json
import subprocess
import textgrad as tg
from dotenv import load_dotenv
from openai import OpenAI
import re
import subprocess


# ============================================================
# 1. Environment Setup
# ============================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

engine = tg.get_engine("gpt-4o-mini")
tg.set_backward_engine(engine, override=True)

CODE_DIR = "temp_code"
os.makedirs(CODE_DIR, exist_ok=True)


# ============================================================
# 2. Utility Functions
# ============================================================
def clean_code_output(raw_code: str) -> str:
    """Extract clean Python code from Markdown or raw text."""
    if "```" in raw_code:
        parts = raw_code.split("```")
        for part in parts:
            if part.strip().startswith("python") or not part.strip().startswith(("```", "")):
                raw_code = part
                break
        raw_code = raw_code.lstrip("python\n").strip()
    return raw_code


def generate_code(prompt: str) -> str:
    """Generate Python code using GPT."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    code = response.choices[0].message.content
    return clean_code_output(code)


def save_and_run(code: str, filename: str = "temp_exec.py") -> str:
    """Save code to file, execute it, and auto-install missing packages if needed."""
    code = clean_code_output(code)
    file_path = os.path.join(CODE_DIR, filename)

    with open(file_path, "w") as f:
        f.write(code)

    def run_script():
        try:
            result = subprocess.run(
                ["python", file_path],
                text=True,
                capture_output=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"ERROR: {e.stderr}"

    # First run
    output = run_script()

    # Check for missing module errors and auto-install
    missing_module_pattern = r"No module named ['\"]([\w\-\.]+)['\"]"
    match = re.search(missing_module_pattern, output)
    if match:
        missing_pkg = match.group(1)
        print(f"⚠️ Detected missing package: {missing_pkg} → installing...")
        try:
            subprocess.run(["pip", "install", missing_pkg], check=True)
            print(f"✅ Installed {missing_pkg}, retrying execution...\n")
            output = run_script()
        except Exception as install_err:
            output += f"\nERROR: Failed to install {missing_pkg}: {install_err}"

    return output


def run_or_fix_code(prompt: str, code: str, max_attempts: int = 3) -> str:
    """Try executing the code; if it fails, ask GPT to fix until success."""
    for attempt in range(1, max_attempts + 1):
        output = save_and_run(code)
        if not output.startswith("ERROR"):
            return code
        print(f"❌ Attempt {attempt} failed, auto-fixing...")
        fix_prompt = (
            f"The following code has an error:\n\n{code}\n\n"
            f"Error message:\n{output}\n\n"
            f"Original instruction:\n{prompt}\n\n"
            "Please fix the code and return ONLY the corrected Python code."
        )
        code = generate_code(fix_prompt)
    return clean_code_output(code)


# ============================================================
# 3. TextGrad Loss & Optimizer
# ============================================================
EVAL_PROMPT = tg.Variable(
    "You are a helpful assistant that evaluates if the provided Python code correctly solves the given problem. "
    "Respond with 'Correct' if the code meets the requirements, otherwise respond with 'Incorrect'.",
    requires_grad=False,
    role_description="system prompt for evaluation"
)

fields = {"problem": None, "dataset": None, "code": None}
format_string = """
Problem: {problem}
Dataset Info: {dataset}
Current Code: {code}
"""

formatted_llm_call = tg.autograd.FormattedLLMCall(
    engine=engine,
    format_string=format_string,
    fields=fields,
    system_prompt=EVAL_PROMPT
)

def loss_fn(problem: tg.Variable, dataset: tg.Variable, code: tg.Variable) -> tg.Variable:
    """Use TextGrad to evaluate code correctness."""
    inputs = {"problem": problem, "dataset": dataset, "code": code}
    return formatted_llm_call(inputs=inputs,
                              response_role_description=f"evaluation of {code.get_role_description()}")


optimizer = tg.TextualGradientDescent(
    engine=engine,
    parameters=[],
    constraints=[
        "Always return valid Python code inside ```python ...```",
        "Always wrap the improved code inside <IMPROVED_VARIABLE> ... </IMPROVED_VARIABLE> tags, and nothing else."
    ]
)


# ============================================================
# 4. Optimization Loop
# ============================================================
def optimize_code(prompt_var: tg.Variable, dataset: tg.Variable, max_steps: int = 3):
    """Main TextGrad optimization loop."""
    initial_prompt = f"""
You are given a dataset description and a task. 
Generate a Python program that accomplishes the task using this dataset.

Task: {prompt_var.value}
Dataset: {dataset.value}
"""
    initial_code = generate_code(initial_prompt)
    code_var = tg.Variable(initial_code, requires_grad=True, role_description="generated python code")
    optimizer.parameters = [code_var]

    
    for step in range(max_steps):
        print(f"\n=== Optimization Step {step + 1} ===")
        print("🧠 Current code preview:\n", code_var.value[:300], "...\n")

        # Try execution or auto-fix
        code_var.set_value(run_or_fix_code(prompt_var.value, clean_code_output(code_var.value)))
        output = save_and_run(clean_code_output(code_var.value))
        print("⚙️ Execution Output:\n", output)

        # Evaluate correctness
        loss = loss_fn(prompt_var, dataset, code_var)
        print("📊 Evaluation result:\n", loss.value)
        loss.backward()
        optimizer.step()

        print("✅ Updated code preview:\n", code_var.value[:300], "...\n")

    return code_var.value


# ============================================================
# 5. Main Configuration
# ============================================================
if __name__ == "__main__":
    # ========= 🧩 CONFIG =========
    TASK_PROMPT = tg.Variable(
        "Write a Python program that loads the dataset, preprocesses it, "
        "and choose top 3 model that you think is the most suitable and trains them to predict the label"
        "output the model that perfect the best"
        "also write code to visualize data with umap"
        "also choose the metrics yourself that express the most comprehensive evaluation of the model performance"
        "also visualize data with the plot you think is the most suitable"
        "output all the plot, metrics with jpg or png file"
        "you have to generate out the plot that is commonly used in thesis and paper, try to make it comprehensive"
        "put all of the result in a folder named results"
        ,
        requires_grad=False,
        role_description="system prompt (task definition)"
    )

    DATASET_PATH = "data_manifest.json"
    MAX_STEPS = 2

    dataset_info = tg.Variable(
        json.dumps(json.load(open(DATASET_PATH)), indent=2),
        requires_grad=False,
        role_description="dataset metadata"
    )

    # ========= 🚀 RUN =========
    final_code = optimize_code(
        prompt_var=TASK_PROMPT,
        dataset=dataset_info,
        max_steps=MAX_STEPS
    )

    save_and_run(clean_code_output(final_code), filename="final_optimized_code.py")

    print("\n=== 🧩 Final Optimized Code ===\n", final_code[:800])
    print("\nSystem prompt used:\n", TASK_PROMPT.value)
