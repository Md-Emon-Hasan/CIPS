import subprocess
import os
import sys
import time
import threading


def run_backend():
    print("Starting Backend (FastAPI)...")
    # Run uvicorn from the root directory, pointing to backend.app.main:app
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
        check=True,
    )


def run_frontend():
    print("Starting Frontend (Vite)...")
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    # Use 'npm.cmd' on Windows
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    subprocess.run([npm_cmd, "run", "dev"], cwd=frontend_dir, check=True)


if __name__ == "__main__":
    try:
        # Start backend in a separate thread/process
        backend_thread = threading.Thread(target=run_backend)
        backend_thread.daemon = True
        backend_thread.start()

        # Give backend a moment to start
        time.sleep(2)

        # Start frontend in the main thread (or separate if needed, but usually fine)
        run_frontend()

    except KeyboardInterrupt:
        print("\nStopping services...")
        sys.exit(0)
