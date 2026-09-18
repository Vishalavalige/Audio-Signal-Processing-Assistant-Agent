"""
launcher.py
-----------
Standalone desktop launcher for the Audio Signal Processing Agent dashboard.

- Starts the Flask server on a free port
- Opens the dashboard in the default browser automatically
- Shows a system-tray-style console window with a Stop button
- Exiting the window shuts down the server cleanly
"""

from __future__ import annotations

import os
import sys
import socket
import threading
import webbrowser
import time
import tkinter as tk
from tkinter import ttk, messagebox

# ── Make sure local modules are importable when running as .exe ──────────────
if getattr(sys, "frozen", False):
    base = sys._MEIPASS
    sys.path.insert(0, base)
    os.chdir(base)

import generate_test_tones
import feature_extractor      # noqa: F401 — ensure bundled
import classifier              # noqa: F401 — ensure bundled


# ---------------------------------------------------------------------------
# Find a free port
# ---------------------------------------------------------------------------

def _free_port(preferred: int = 5000) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


# ---------------------------------------------------------------------------
# Flask server (runs in a daemon thread)
# ---------------------------------------------------------------------------

def _start_server(port: int) -> None:
    # Import here so PyInstaller picks up all Flask internals
    from app import app
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Tkinter control window
# ---------------------------------------------------------------------------

class ControlWindow:
    def __init__(self, port: int) -> None:
        self.port = port
        self.url  = f"http://127.0.0.1:{port}"

        self.root = tk.Tk()
        self.root.title("Audio Signal Processing Agent")
        self.root.geometry("420x260")
        self.root.resizable(False, False)
        self.root.configure(bg="#1f2328")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self.root

        # Title bar area
        tk.Label(root, text="🎵", font=("Segoe UI Emoji", 28),
                 bg="#1f2328", fg="#ffffff").pack(pady=(22, 4))
        tk.Label(root, text="Audio Signal Processing Agent",
                 font=("Segoe UI", 13, "bold"),
                 bg="#1f2328", fg="#ffffff").pack()
        tk.Label(root, text="Problem Statement 32 — Acoustic event detection",
                 font=("Segoe UI", 9), bg="#1f2328", fg="#8b949e").pack(pady=(2, 14))

        # Status indicator
        self._status_var = tk.StringVar(value="⏳  Starting server…")
        tk.Label(root, textvariable=self._status_var,
                 font=("Segoe UI", 9), bg="#1f2328", fg="#f0f2f5").pack()

        # Progress bar (indeterminate while starting)
        self._pb = ttk.Progressbar(root, mode="indeterminate", length=300)
        self._pb.pack(pady=(8, 14))
        self._pb.start(12)

        # Buttons
        btn_frame = tk.Frame(root, bg="#1f2328")
        btn_frame.pack()

        self._open_btn = tk.Button(
            btn_frame, text="Open Dashboard",
            font=("Segoe UI", 10, "bold"),
            bg="#3b82d4", fg="#ffffff", relief="flat",
            padx=16, pady=6, cursor="hand2",
            command=self._open_browser
        )
        self._open_btn.pack(side="left", padx=6)

        tk.Button(
            btn_frame, text="Stop & Exit",
            font=("Segoe UI", 10),
            bg="#e5484d", fg="#ffffff", relief="flat",
            padx=16, pady=6, cursor="hand2",
            command=self._on_close
        ).pack(side="left", padx=6)

        tk.Label(root, text=self.url,
                 font=("Consolas", 9), bg="#1f2328", fg="#57606a").pack(pady=(12, 0))

    # ── Actions ─────────────────────────────────────────────────────────────

    def _open_browser(self) -> None:
        webbrowser.open(self.url)

    def _on_close(self) -> None:
        if messagebox.askokcancel("Quit", "Stop the server and exit?"):
            self.root.destroy()
            os._exit(0)

    def mark_ready(self) -> None:
        """Called from background thread once Flask is accepting connections."""
        self.root.after(0, self._set_ready)

    def _set_ready(self) -> None:
        self._pb.stop()
        self._pb.configure(mode="determinate", value=100,
                           style="green.Horizontal.TProgressbar")
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor="#2d333b", background="#30a46c")
        self._status_var.set(f"✅  Server running on port {self.port}")

    # ── Run ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Wait until Flask is actually accepting connections
# ---------------------------------------------------------------------------

def _wait_then_open(port: int, window: ControlWindow) -> None:
    for _ in range(40):          # wait up to 4 s
        time.sleep(0.1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break

    window.mark_ready()
    webbrowser.open(f"http://127.0.0.1:{port}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure test tones exist
    sample_dir = generate_test_tones.OUTPUT_DIR
    has_tones  = (
        os.path.isdir(sample_dir)
        and any(f.endswith(".wav") for f in os.listdir(sample_dir))
    )
    if not has_tones:
        generate_test_tones.main()

    port = _free_port(5000)

    # Start Flask in background daemon thread
    t = threading.Thread(target=_start_server, args=(port,), daemon=True)
    t.start()

    # Build control window
    win = ControlWindow(port)

    # Wait for server + open browser (in another thread so UI stays responsive)
    threading.Thread(target=_wait_then_open, args=(port, win), daemon=True).start()

    # Run Tkinter main loop (blocks until window closed)
    win.run()


if __name__ == "__main__":
    main()
