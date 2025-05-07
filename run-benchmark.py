import queue
import subprocess
import pathlib
import threading
import time
import sys
import logging
from datetime import datetime

HEF_DIR = pathlib.Path("models/object-detection")
OUTPUT_LOG = "benchmark_results.log"
VIDEO_MP4_PATH = "samples/videos/20200229174849.mp4"
TIMEOUT_SECONDS = 120  # 2 分鐘

def find_hef_files():
    return sorted(HEF_DIR.glob("**/hailo-8l-hef/*.hef"))

def print_progress(percent: float, last_line: str, bar_len: int = 30):
    filled_len = int(round(bar_len * percent / 100))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(f"\r[{bar}] {percent:.1f}%\n{last_line}\n")
    sys.stdout.flush()

def run_inference(hef_path: pathlib.Path) -> str:
    cmd = [
        "python3",
        "./src/main.py",
        "-c=10",
        "-t=10",
        "-f",
        f"-spath={VIDEO_MP4_PATH}",
        f"-m={hef_path}"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    print(f"run {cmd}")

    output_lines = []
    output_queue = queue.Queue()

    def reader_thread():
        for line in proc.stdout:  # type: ignore
            output_queue.put(line)

    thread = threading.Thread(target=reader_thread, daemon=True)
    thread.start()

    start_time = time.time()
    last_line = ""
    try:
        while True:
            try:
                # timeout 設為 1 秒等待輸出
                line = output_queue.get()
                line = line.strip()
                output_lines.append(line)
                last_line = line
            except queue.Empty:
                pass  # 沒輸出，進入下一輪檢查

            elapsed = time.time() - start_time
            percent = min(elapsed / TIMEOUT_SECONDS * 100, 100)
            print_progress(percent, last_line)

            if proc.poll() is not None:  # process 結束
                break
            if elapsed > TIMEOUT_SECONDS:
                proc.kill()
                print_progress(100, "[Timeout]")
                return f"{hef_path.name}: [Timeout after {TIMEOUT_SECONDS}s]"

        last_line = output_lines[-1] if output_lines else "[No output]"
        print_progress(100, last_line)
        return f"{hef_path.name}: {last_line}"
    except Exception as e:
        proc.kill()
        print_progress(100, f"[Error: {e}]")
        return f"{hef_path.name}: [Error: {e}]"

def main():
    hef_files = find_hef_files()
    with open(OUTPUT_LOG, "w") as log_file:
        for hef_path in hef_files:
            print(f"\nRunning {hef_path.name}")
            result = run_inference(hef_path)
            log_file.write(result + "\n")
        print()  # newline after last progress bar

if __name__ == "__main__":
    main()
