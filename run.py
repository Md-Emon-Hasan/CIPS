import subprocess
import os
import sys
import time
import threading


def run_backend() -> None:
    print("Starting Backend (FastAPI)...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--reload",
                "--port",
                "8000",
            ],
            check=False,
        )
    except Exception as e:
        print(f"Backend failed: {e}")


def run_frontend() -> None:
    print("Starting Frontend (Vite)...")
    frontend_dir: str = os.path.abspath("frontend")
    npm: str = "npm.cmd" if os.name == "nt" else "npm"

    # Auto install if node_modules is missing
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("Installing frontend dependencies...")
        subprocess.run([npm, "install"], cwd=frontend_dir, check=True)

    subprocess.run([npm, "run", "dev"], cwd=frontend_dir, check=False)


if __name__ == "__main__":
    # Start backend in background
    threading.Thread(target=run_backend, daemon=True).start()

    # Wait for backend to warm up
    time.sleep(2)

    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\nShutting down...")
