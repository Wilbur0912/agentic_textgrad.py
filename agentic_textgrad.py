import os, subprocess
import textgrad as tg
from dotenv import load_dotenv
from openai import OpenAI

# ===============================
# Setup
# ===============================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

engine = tg.get_engine("gpt-4o")
tg.set_backward_engine(engine, override=True)

system_prompt = tg.Variable(
    "You are a Python assistant that writes sklearn ML code.",
    requires_grad=False,
    role_description="system prompt"
)
generator = tg.BlackboxLLM(engine, system_prompt)

optimizer = tg.TextualGradientDescent(
    engine=engine,
    parameters=[],
    constraints=["Always return valid Python code inside ```python ...```"]
)

CODE_DIR = "temp_code"
os.makedirs(CODE_DIR, exist_ok=True)


# ===============================
# Utility Functions
# ===============================
def clean_code_output(raw_code: str) -> str:
    if "```" in raw_code:
        parts = raw_code.split("```")
        for part in parts:
            if part.strip().startswith("python") or not part.strip().startswith(("```", "")):
                raw_code = part
                break
        raw_code = raw_code.lstrip("python\n").strip()
    return raw_code

def generate_code(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    code = response.choices[0].message.content
    return clean_code_output(code)

def save_and_run(code: str, filename: str = "temp_exec.py") -> str:
    """存檔並執行程式碼，返回 stdout 或錯誤"""
    file_path = os.path.join(CODE_DIR, filename)
    with open(file_path, "w") as f:
        f.write(code)
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

def run_or_fix_code(task: str, code: str, max_attempts: int = 3) -> str:
    """
    嘗試執行程式碼；若報錯，呼叫 GPT 修復，直到成功或達到 max_attempts。
    """
    for attempt in range(1, max_attempts + 1):
        output = save_and_run(code)
        if not output.startswith("ERROR"):
            return code  # 成功執行
        print(f"❌ Attempt {attempt} failed, auto-fixing...")
        fix_prompt = (
            f"The following code has an error:\n\n{code}\n\n"
            f"Error message:\n{output}\n\n"
            "Please fix the code and return ONLY the corrected Python code."
        )
        code = generate_code(fix_prompt)
    return code  # 可能最後還是有錯，但已嘗試修復


# ===============================
# TextGrad Pipeline
# ===============================
def evaluator(output: str, expected: str, code_var: tg.Variable) -> tg.Variable:
    """比對 runner output 和預期結果"""
    if output.strip() == expected.strip():
        value = "Correct"
    else:
        value = "Incorrect"
    return tg.Variable(
        value,
        requires_grad=True,
        role_description="evaluation of generated code",
        predecessors=[code_var]
    )

def optimize_code(task: str, expected_output: str, max_steps: int = 3):
    # 初始 code 生成 
    initial_code = generate_code(task)
    code_var = tg.Variable(initial_code, requires_grad=True, role_description="generated python code")
    optimizer.parameters = [code_var]

    for step in range(max_steps):
        print(f"\n=== Optimization Step {step} ===")
        print("Current code:\n", code_var.value[:300], "...\n")

        # 嘗試執行，若失敗則修復
        code_var.set_value(run_or_fix_code(task, code_var.value))
        output = save_and_run(code_var.value)
        print("Execution Output:\n", output)

        # Evaluator → loss variable
        eval_var = evaluator(output, expected_output, code_var)

        # Backward
        total_loss = tg.sum([eval_var])
        total_loss.backward()

        # Optimizer 更新 code
        optimizer.step()
        print("Updated code:\n", code_var.value[:300], "...\n")

    return code_var.value


# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    final_code = optimize_code(
        "Write a Python function that prints 'hello world'.",
        expected_output="hello world",
        max_steps=2
    )
    print("\n=== Final Optimized Code ===\n", final_code[:500])
