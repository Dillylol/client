
import json
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from data_stream import DataStream

class DevControllerApp(tk.Tk):
    """
    Two-tab controller:
      - Telemetry: live key/value view that adapts to whatever the bot sends
      - Terminal: send commands and view responses
    """
    def __init__(self, stream: DataStream | None = None):
        super().__init__()
        self.title("JULES Dev Controller")
        self.geometry("900x600")

        self.stream = stream or DataStream()
        self.update_interval = 0.1  # seconds
        self._last_update_ts = 0.0

        self._build_ui()
        self._schedule_tick()

    # ---------- UI ----------
    def _build_ui(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Telemetry tab
        self.telemetry_frame = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.telemetry_frame, text="Telemetry")

        self.telemetry_canvas = tk.Canvas(self.telemetry_frame, highlightthickness=0)
        self.telemetry_scroll = ttk.Scrollbar(self.telemetry_frame, orient="vertical", command=self.telemetry_canvas.yview)
        self.telemetry_inner = ttk.Frame(self.telemetry_canvas)
        self.telemetry_inner.bind(
            "<Configure>",
            lambda e: self.telemetry_canvas.configure(scrollregion=self.telemetry_canvas.bbox("all"))
        )
        self.telemetry_canvas.create_window((0, 0), window=self.telemetry_inner, anchor="nw")
        self.telemetry_canvas.configure(yscrollcommand=self.telemetry_scroll.set)

        self.telemetry_canvas.grid(row=0, column=0, sticky="nsew")
        self.telemetry_scroll.grid(row=0, column=1, sticky="ns")
        self.telemetry_frame.rowconfigure(0, weight=1)
        self.telemetry_frame.columnconfigure(0, weight=1)

        self.telemetry_labels: dict[str, ttk.Label] = {}

        # Terminal tab
        self.terminal_frame = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.terminal_frame, text="Terminal")

        self.terminal_output = scrolledtext.ScrolledText(self.terminal_frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.terminal_output.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=(0, 8))

        self.cmd_var = tk.StringVar()
        self.cmd_entry = ttk.Entry(self.terminal_frame, textvariable=self.cmd_var)
        self.cmd_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.cmd_entry.bind("<Return>", lambda e: self._send_cmd())

        self.send_btn = ttk.Button(self.terminal_frame, text="Send", command=self._send_cmd)
        self.send_btn.grid(row=1, column=1, sticky="ew")

        # Quick actions (customize as needed)
        self.quick_frame = ttk.Frame(self.terminal_frame)
        self.quick_frame.grid(row=1, column=2, columnspan=2, sticky="e")

        for label, cmd in [
            ("Ping", "ping"),
            ("Zero Odom", "zero_odom"),
            ("Stop", "stop"),
            ("Reboot", "reboot"),
        ]:
            ttk.Button(self.quick_frame, text=label, command=lambda c=cmd: self._send_cmd(c)).pack(side=tk.LEFT, padx=4)

        # resize behavior
        self.terminal_frame.columnconfigure(0, weight=1)
        self.terminal_frame.columnconfigure(1, weight=0)
        self.terminal_frame.columnconfigure(2, weight=0)
        self.terminal_frame.columnconfigure(3, weight=0)
        self.terminal_frame.rowconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ---------- Telemetry loop ----------
    def _schedule_tick(self):
        self.after(50, self._tick)

    def _tick(self):
        now = time.time()
        if now - self._last_update_ts >= self.update_interval:
            self._last_update_ts = now
            self._update_telemetry()
        self._schedule_tick()

    def _update_telemetry(self):
        data = self.stream.get_data()
        if not data:
            self.status_var.set("No data (check connection)")
            return

        latest = data[-1]
        attrs = latest.__dict__.copy()
        # merge in extra fields if present
        extra = attrs.pop("_extra", {}) or {}
        attrs.update(extra)

        # Remove labels that are no longer present
        current_keys = set(self.telemetry_labels.keys())
        new_keys = set(attrs.keys())
        for k in current_keys - new_keys:
            self.telemetry_labels[k].destroy()
            del self.telemetry_labels[k]

        # Update/create labels
        for row_idx, key in enumerate(sorted(attrs.keys())):
            val = attrs[key]
            if isinstance(val, float):
                disp = f"{val:.3f}"
            else:
                disp = str(val)

            if key not in self.telemetry_labels:
                self.telemetry_labels[key] = ttk.Label(self.telemetry_inner, text=f"{key}: {disp}")
                self.telemetry_labels[key].grid(row=row_idx, column=0, sticky="w", pady=2)
            else:
                self.telemetry_labels[key]["text"] = f"{key}: {disp}"

        self.status_var.set(f"Updated: {time.strftime('%H:%M:%S')} (items: {len(attrs)})")

    # ---------- Terminal ----------
    def _append_terminal(self, text: str):
        self.terminal_output.configure(state=tk.NORMAL)
        self.terminal_output.insert(tk.END, text + "\n")
        self.terminal_output.configure(state=tk.DISABLED)
        self.terminal_output.see(tk.END)

    def _send_cmd(self, preset: str | None = None):
        cmd = preset if preset is not None else self.cmd_var.get().strip()
        if not cmd:
            return
        self._append_terminal(f"> {cmd}")
        self.cmd_var.set("")

        try:
            result = self.stream.send_command(cmd)
            if result.get("ok") is False:
                self._append_terminal(f"[ERR] {json.dumps(result, ensure_ascii=False)}")
            else:
                # pretty print if dict-like, else just str
                if isinstance(result, dict):
                    pretty = json.dumps(result, indent=2, ensure_ascii=False)
                    self._append_terminal(pretty)
                else:
                    self._append_terminal(str(result))
        except Exception as e:
            self._append_terminal(f"[EXC] {e!r}")
