import os
import sys
import time
import subprocess
import threading


def setup_venv():
    venv = os.path.abspath(".venv")
    py = (
        os.path.join(venv, "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(venv, "bin", "python")
    )

    if not os.path.exists(venv):
        print(">>> Creating venv...")
        subprocess.run([sys.executable, "-m", "venv", venv])

    reqs = os.path.join("backend", "requirements.txt")
    if os.path.exists(reqs):
        print(">>> Installing backend packages...")
        subprocess.run([py, "-m", "pip", "install", "-r", reqs])

    return py


def run_backend(py):
    print(">>> Starting Backend...")
    try:
        subprocess.run(
            [py, "-m", "uvicorn", "backend.app.main:app", "--reload", "--port", "8000"]
        )
    except Exception as e:
        print(f"Error: {e}")


def setup_frontend():
    ui_dir = os.path.abspath("frontend")
    npm = "npm.cmd" if os.name == "nt" else "npm"

    if not os.path.exists(os.path.join(ui_dir, "node_modules")):
        print(">>> Installing frontend dependencies...")
        subprocess.run([npm, "install"], cwd=ui_dir)
    return npm, ui_dir


if __name__ == "__main__":
    # 1. Setup Backend (Venv + Pip)
    python_path = setup_venv()

    # 2. Setup Frontend (Npm Install)
    npm_cmd, frontend_dir = setup_frontend()

    # 3. Start Backend in background thread
    backend_thread = threading.Thread(
        target=run_backend, args=(python_path,), daemon=True
    )
    backend_thread.start()

    # Give backend a moment to initialize
    time.sleep(1)

    # 4. Start Frontend in main thread
    print(">>> Starting Frontend...")
    try:
        subprocess.run([npm_cmd, "run", "dev"], cwd=frontend_dir)
    except KeyboardInterrupt:
        print("\nStopping...")
