from __future__ import annotations

import json
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from data_stream import DataStream
import codec

FILTER_TAGS = [
    ("pose", "Pose"),
    ("motors", "Motors"),
    ("imu", "IMU"),
    ("power", "Power"),
    ("debug", "Debug"),
    ("custom", "Custom"),
]

BATTERY_COLORS = [
    (12.0, "#64dd17"),
    (11.5, "#fbc02d"),
    (0.0, "#ef5350"),
]


class DevControllerApp(tk.Tk):
    """IDE-style dashboard for the Jules FTC bridge."""

    def __init__(self, stream: Optional[DataStream] = None):
        super().__init__()
        self.title("JULES Dev Controller")
        self.geometry("1280x780")
        self.configure(bg="#1f1f1f")

        self.stream = stream or DataStream()

        self.update_interval = 0.2  # seconds
        self._last_update_ts = 0.0
        self._dropped_polls = 0

        self.telemetry_history: Deque[Dict[str, Any]] = deque(maxlen=400)
        self._arrival_times: Deque[float] = deque(maxlen=400)
        self.command_history: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.history_details: Dict[str, Dict[str, Any]] = {}
        self.saved_commands: List[Dict[str, Any]] = [
            {"label": "Drive Forward", "name": "drive", "args": {"t": 0.5, "p": 0.5, "duration_ms": 400}},
            {"label": "Strafe Right", "name": "strafe", "args": {"speed": 0.4, "duration_ms": 400}},
            {"label": "Turn Left", "name": "turn", "args": {"speed": -0.4, "degrees": 30}},
            {"label": "Stop", "name": "stop", "args": {}},
            {"label": "Ping", "type": codec.PING_TYPE, "payload": {"source": "client"}},
        ]

        self.status_vars: Dict[str, tk.StringVar] = {}
        self.status_value_labels: Dict[str, ttk.Label] = {}
        self.status_actual: Dict[str, Any] = {}
        self.status_masked_keys: set[str] = {"token"}
        self._token_visible = False

        self.connection_state_var = tk.StringVar(value="DISCONNECTED")
        self.status_bar_var = tk.StringVar(value="Ready")
        self.base_url_var = tk.StringVar(value=self.stream.base_url)
        self.token_var = tk.StringVar(value=self.stream.token or "")
        self.token_toggle_text = tk.StringVar(value="Show")
        self.token_entry: Optional[ttk.Entry] = None

        self.filter_vars: Dict[str, tk.BooleanVar] = {}

        self._build_ui()
        self._update_masked_token()
        self._schedule_tick()

    # ------------------------------------------------------------------
    # UI construction
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#1f1f1f")
        style.configure("App.TLabel", background="#1f1f1f", foreground="#f5f5f5", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1f1f1f", foreground="#9cdcfe", font=("Segoe UI", 11, "bold"))
        style.configure("StatusValue.TLabel", background="#1f1f1f", foreground="#dcdcdc", font=("Segoe UI", 10, "bold"))
        style.configure("StatusHeader.TLabel", background="#1f1f1f", foreground="#bbbbbb", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("App.TCheckbutton", background="#1f1f1f", foreground="#f5f5f5")
        style.map("App.TCheckbutton",
                  background=[("selected", "#1f1f1f"), ("!selected", "#1f1f1f")],
                  foreground=[("selected", "#9cdcfe"), ("!selected", "#f5f5f5")])
        style.configure("Dark.TEntry", fieldbackground="#262626", foreground="#f5f5f5", insertcolor="#f5f5f5")
        style.configure("Treeview", background="#262626", fieldbackground="#262626", foreground="#e0e0e0")
        style.configure("Treeview.Heading", background="#333333", foreground="#f0f0f0")
        style.map("Accent.TButton", background=[("active", "#0e639c"), ("!active", "#007acc")])

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_pane.grid(row=0, column=0, sticky="nsew")

        self.status_panel = self._build_status_panel(main_pane)
        main_pane.add(self.status_panel, weight=0)

        right_pane = ttk.Panedwindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane, weight=1)

        self.notebook_frame = ttk.Frame(right_pane, padding=12, style="App.TFrame")
        right_pane.add(self.notebook_frame, weight=3)

        self.nb = ttk.Notebook(self.notebook_frame)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.data_flow_tab = ttk.Frame(self.nb, padding=12, style="App.TFrame")
        self.nb.add(self.data_flow_tab, text="Data Flow")
        self._build_data_flow_tab(self.data_flow_tab)

        self.requests_tab = ttk.Frame(self.nb, padding=12, style="App.TFrame")
        self.nb.add(self.requests_tab, text="Requests")
        self._build_requests_tab(self.requests_tab)

        self.console_pane = ttk.Panedwindow(right_pane, orient=tk.HORIZONTAL)
        right_pane.add(self.console_pane, weight=2)
        self._build_console(self.console_pane)

        status_bar = ttk.Label(self, textvariable=self.status_bar_var, anchor="w", style="App.TLabel")
        status_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

    def _build_status_panel(self, parent: ttk.Panedwindow) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=16, style="App.TFrame")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Connection", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        conn_row = ttk.Frame(frame, style="App.TFrame")
        conn_row.grid(row=1, column=0, sticky="ew", pady=(6, 12))
        ttk.Label(conn_row, textvariable=self.connection_state_var, style="StatusValue.TLabel").pack(side=tk.LEFT)

        ttk.Label(frame, text="Robot Base URL", style="StatusHeader.TLabel").grid(row=2, column=0, sticky="w")
        url_entry = ttk.Entry(frame, textvariable=self.base_url_var, style="Dark.TEntry")
        url_entry.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(frame, text="Token", style="StatusHeader.TLabel").grid(row=4, column=0, sticky="w")
        token_row = ttk.Frame(frame, style="App.TFrame")
        token_row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.token_entry = ttk.Entry(token_row, textvariable=self.token_var, style="Dark.TEntry", show="•")
        self.token_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        toggle_btn = ttk.Button(token_row, textvariable=self.token_toggle_text, command=self._toggle_token_visibility)
        toggle_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(token_row, text="Copy", command=lambda: self._copy_to_clipboard(self.token_var.get())).pack(side=tk.LEFT, padx=(8, 0))

        btn_row = ttk.Frame(frame, style="App.TFrame")
        btn_row.grid(row=6, column=0, sticky="ew", pady=(0, 16))
        connect_btn = ttk.Button(btn_row, text="Connect", command=self._on_connect, style="Accent.TButton")
        connect_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Copy Base", command=lambda: self._copy_to_clipboard(self.base_url_var.get())).pack(side=tk.LEFT, padx=(8, 0))

        row = 7
        for key, label, copyable in [
            ("battery", "Battery", False),
            ("ping", "Ping", False),
            ("heartbeat", "Heartbeat Age", False),
            ("ip", "Robot IP", True),
            ("opmode", "Active OpMode", False),
            ("token", "Token", True),
            ("rate", "Messages / s", False),
            ("dropped", "Dropped Frames", False),
        ]:
            self._create_status_row(frame, row, key, label, copyable=copyable)
            row += 2

        ttk.Label(frame, text="Quick Actions", style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=(16, 4))
        row += 1
        quick_frame = ttk.Frame(frame, style="App.TFrame")
        quick_frame.grid(row=row, column=0, sticky="ew")
        for cmd in ["ping", "stop", "announce"]:
            ttk.Button(quick_frame, text=cmd.title(), command=lambda c=cmd: self._send_quick_action(c)).pack(side=tk.LEFT, padx=(0, 6))

        return frame

    def _create_status_row(self, frame: ttk.Frame, row: int, key: str, label: str, *, copyable: bool = False) -> None:
        ttk.Label(frame, text=label, style="StatusHeader.TLabel").grid(row=row, column=0, sticky="w")
        value_var = tk.StringVar(value="—")
        value_label = ttk.Label(frame, textvariable=value_var, style="StatusValue.TLabel")
        value_label.grid(row=row + 1, column=0, sticky="w", pady=(0, 6))

        handler = None
        if copyable and key in self.status_masked_keys:
            handler = lambda _e, k=key: self._handle_status_click(k, toggle_mask=True)
        elif copyable:
            handler = lambda _e, k=key: self._handle_status_click(k)
        elif key in self.status_masked_keys:
            handler = lambda _e, k=key: self._handle_status_click(k, toggle_mask=True)

        if handler:
            value_label.configure(cursor="hand2")
            value_label.bind("<Button-1>", handler)

        self.status_vars[key] = value_var
        self.status_value_labels[key] = value_label
        self.status_actual[key] = None

    def _build_data_flow_tab(self, frame: ttk.Frame) -> None:
        filter_frame = ttk.Frame(frame, style="App.TFrame")
        filter_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 12))
        ttk.Label(filter_frame, text="Filters", style="StatusHeader.TLabel").pack(side=tk.LEFT)
        for tag, label in FILTER_TAGS:
            var = tk.BooleanVar(value=True)
            self.filter_vars[tag] = var
            chk = ttk.Checkbutton(filter_frame, text=label, variable=var, command=self._refresh_data_flow, style="App.TCheckbutton")
            chk.pack(side=tk.LEFT, padx=6)

        self.data_flow_text = scrolledtext.ScrolledText(
            frame,
            height=18,
            wrap=tk.WORD,
            state=tk.DISABLED,
            background="#252526",
            foreground="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.data_flow_text.pack(fill=tk.BOTH, expand=True)
        self.data_flow_text.tag_configure("timestamp", foreground="#6a9955")
        self.data_flow_text.tag_configure("telemetry", foreground="#9cdcfe")
        self.data_flow_text.tag_configure("command", foreground="#c586c0")
        self.data_flow_text.tag_configure("ack", foreground="#4fc1ff")

    def _build_requests_tab(self, frame: ttk.Frame) -> None:
        pane = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        library_frame = ttk.Frame(pane, padding=(0, 0, 12, 0), style="App.TFrame")
        pane.add(library_frame, weight=1)
        ttk.Label(library_frame, text="Saved Commands", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.library_list = tk.Listbox(
            library_frame,
            height=10,
            background="#1e1e1e",
            foreground="#d4d4d4",
            selectbackground="#264f78",
            activestyle="none",
        )
        self.library_list.pack(fill=tk.BOTH, expand=True)
        for idx, entry in enumerate(self.saved_commands):
            self.library_list.insert(tk.END, entry["label"])
        self.library_list.bind("<<ListboxSelect>>", self._on_library_select)

        pane_right = ttk.Panedwindow(pane, orient=tk.VERTICAL)
        pane.add(pane_right, weight=3)

        editor_frame = ttk.Frame(pane_right, padding=(0, 0, 0, 8), style="App.TFrame")
        pane_right.add(editor_frame, weight=3)
        ttk.Label(editor_frame, text="Command Editor", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.request_editor = scrolledtext.ScrolledText(
            editor_frame,
            height=12,
            wrap=tk.WORD,
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.request_editor.pack(fill=tk.BOTH, expand=True)
        self.request_editor.bind("<Control-Return>", self._send_request_from_editor)

        button_row = ttk.Frame(editor_frame, style="App.TFrame")
        button_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(button_row, text="Validate", command=self._validate_editor).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Send", command=lambda: self._send_request_from_editor()).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Clear", command=lambda: self.request_editor.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(8, 0))

        history_frame = ttk.Frame(pane_right, style="App.TFrame")
        pane_right.add(history_frame, weight=2)
        ttk.Label(history_frame, text="History", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        columns = ("id", "name", "status", "time")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=6)
        for col in columns:
            self.history_tree.heading(col, text=col.title())
            self.history_tree.column(col, width=120 if col != "name" else 180, anchor="center")
        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_tree.bind("<Double-1>", self._on_history_double_click)

    def _build_console(self, pane: ttk.Panedwindow) -> None:
        input_frame = ttk.Frame(pane, padding=12, style="App.TFrame")
        pane.add(input_frame, weight=1)
        ttk.Label(input_frame, text="Input Terminal", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.console_input = scrolledtext.ScrolledText(
            input_frame,
            height=12,
            wrap=tk.NONE,
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.console_input.pack(fill=tk.BOTH, expand=True)
        self.console_input.bind("<Control-Return>", self._send_console_input)

        console_btns = ttk.Frame(input_frame, style="App.TFrame")
        console_btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(console_btns, text="Send", command=lambda: self._send_console_input()).pack(side=tk.LEFT)
        ttk.Button(console_btns, text="Clear", command=lambda: self.console_input.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(8, 0))

        output_frame = ttk.Frame(pane, padding=12, style="App.TFrame")
        pane.add(output_frame, weight=1)
        ttk.Label(output_frame, text="Output Terminal", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.console_output = scrolledtext.ScrolledText(
            output_frame,
            height=12,
            wrap=tk.WORD,
            state=tk.DISABLED,
            background="#1e1e1e",
            foreground="#c7c7c7",
            insertbackground="#d4d4d4",
        )
        self.console_output.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Connection helpers
    def _on_connect(self) -> None:
        base = self.base_url_var.get().strip()
        if not base:
            messagebox.showerror("Invalid URL", "Please enter the robot base URL")
            return
        token = self.token_var.get().strip()
        self.stream.connect(base, token=token or None)
        self.base_url_var.set(self.stream.base_url)
        if token:
            self.status_actual["token"] = token
        else:
            self.status_actual["token"] = None
        self._update_masked_token()
        self.connection_state_var.set("CONNECTING…")
        self.status_bar_var.set(f"Connecting to {base}")
        self.telemetry_history.clear()
        self._arrival_times.clear()
        self._dropped_polls = 0
        self.data_flow_text.configure(state=tk.NORMAL)
        self.data_flow_text.delete("1.0", tk.END)
        self.data_flow_text.configure(state=tk.DISABLED)

    def _copy_to_clipboard(self, text: str) -> None:
        if not text:
            self.status_bar_var.set("Nothing to copy")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_bar_var.set("Copied to clipboard")

    def _handle_status_click(self, key: str, *, toggle_mask: bool = False) -> None:
        actual = self.status_actual.get(key)
        if actual:
            self._copy_to_clipboard(str(actual))
        if toggle_mask:
            self._toggle_token_visibility()

    # ------------------------------------------------------------------
    # Data flow and telemetry
    def _schedule_tick(self) -> None:
        self.after(50, self._tick)

    def _tick(self) -> None:
        now = time.time()
        if now - self._last_update_ts >= self.update_interval:
            self._last_update_ts = now
            self._update_telemetry()
        self._schedule_tick()

    def _update_telemetry(self) -> None:
        snapshot = self.stream.connection_snapshot()
        data = self.stream.get_data()
        if not data:
            self._dropped_polls += 1
            state_text = self._state_text(snapshot)
            self.connection_state_var.set(state_text)
            if snapshot.get("last_error"):
                self.status_bar_var.set(f"Error: {snapshot['last_error']}")
            elif snapshot.get("state") == "CONNECTING":
                target = snapshot.get("base_url") or self.base_url_var.get()
                self.status_bar_var.set(f"Connecting to {target}…")
            elif snapshot.get("state") == "RECONNECTING":
                attempt = snapshot.get("reconnect_attempts", 0)
                suffix = f" (attempt {attempt})" if attempt else ""
                self.status_bar_var.set(f"Reconnecting{suffix}…")
            else:
                self.status_bar_var.set("Waiting for telemetry…")
            self._update_status_panel(None)
            return

        self.connection_state_var.set("CONNECTED")
        new_entries: List[Dict[str, Any]] = []
        for item in data:
            entry = self._normalize_entry(item)
            if not self.stream.is_new_payload(entry):
                continue
            payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else entry
            entry["_tags"] = self._extract_tags(payload)
            self.telemetry_history.append(entry)
            self._arrival_times.append(time.time())
            new_entries.append(entry)

        if new_entries:
            for entry in new_entries:
                self._refresh_data_flow(new_entry=entry)
            latest = new_entries[-1]
            self._update_status_panel(latest)
            self.status_bar_var.set(f"Received {len(new_entries)} message(s)")
        else:
            latest = self.telemetry_history[-1] if self.telemetry_history else None
            self._update_status_panel(latest)
            self.status_bar_var.set("No new telemetry (duplicate)")

    def _state_text(self, snapshot: Dict[str, Any]) -> str:
        state = snapshot.get("state", "DISCONNECTED")
        if state == "CONNECTED":
            return "CONNECTED"
        if state == "CONNECTING":
            return "CONNECTING…"
        if state == "RECONNECTING":
            attempts = snapshot.get("reconnect_attempts", 0)
            if attempts:
                return f"RECONNECTING ({attempts})"
            return "RECONNECTING…"
        if state == "ERROR":
            return "ERROR"
        return "DISCONNECTED"

    def _normalize_entry(self, entry: Any) -> Dict[str, Any]:
        if hasattr(entry, "to_dict") and callable(getattr(entry, "to_dict")):
            payload = entry.to_dict()
            return {
                "type": codec.TELEMETRY_TYPE,
                "payload": payload,
                "ts": payload.get("ts") or payload.get("t") or int(time.time() * 1000),
            }
        if isinstance(entry, dict):
            normalized = dict(entry)
            normalized.setdefault("type", codec.TELEMETRY_TYPE)
            normalized.setdefault("ts", int(time.time() * 1000))
            return normalized
        return {
            "type": "event",
            "payload": {"value": entry},
            "ts": int(time.time() * 1000),
        }

    def _extract_tags(self, payload: Dict[str, Any]) -> List[str]:
        tags = set(payload.get("tags", []))
        if any(k in payload for k in ("pose", "x", "y", "heading")):
            tags.add("pose")
        if any(k in payload for k in ("drive", "wheel", "motor")):
            tags.add("motors")
        if any(k in payload for k in ("imu", "pitch", "roll", "yaw")):
            tags.add("imu")
        if any(k in payload for k in ("battery", "battery_V", "battery_v", "loop_ms")):
            tags.add("power")
        if not tags:
            tags.add("custom")
        return sorted(tags)

    def _refresh_data_flow(self, *, full: bool = False, new_entry: Optional[Dict[str, Any]] = None) -> None:
        include = {tag for tag, var in self.filter_vars.items() if var.get()}

        def matches(entry: Dict[str, Any]) -> bool:
            if not include:
                return False
            entry_tags = set(entry.get("_tags", []))
            return bool(include.intersection(entry_tags))

        if full or new_entry is None:
            self.data_flow_text.configure(state=tk.NORMAL)
            self.data_flow_text.delete("1.0", tk.END)
            for entry in self.telemetry_history:
                if matches(entry):
                    self._append_data_flow_line(entry)
            self.data_flow_text.configure(state=tk.DISABLED)
            return

        if matches(new_entry):
            self.data_flow_text.configure(state=tk.NORMAL)
            self._append_data_flow_line(new_entry)
            self.data_flow_text.configure(state=tk.DISABLED)

    def _append_data_flow_line(self, entry: Dict[str, Any]) -> None:
        timestamp = self._format_timestamp(entry)
        entry_type = entry.get("type", codec.TELEMETRY_TYPE)
        payload = entry.get("payload", entry)
        summary = self._summarize_payload(payload)
        line = f"[{timestamp}] {entry_type.upper():<9} {summary}\n"
        self.data_flow_text.insert(tk.END, line)
        start_index = self.data_flow_text.index(f"end-{len(line)}c")
        end_index = self.data_flow_text.index(tk.END)
        timestamp_end = self.data_flow_text.index(f"{start_index}+{len(timestamp) + 2}c")
        self.data_flow_text.tag_add("timestamp", start_index, timestamp_end)
        tag_name = entry_type if entry_type in {codec.TELEMETRY_TYPE, codec.COMMAND_TYPE, codec.ACK_TYPE} else "telemetry"
        self.data_flow_text.tag_add(tag_name, timestamp_end, end_index)
        self.data_flow_text.see(tk.END)

    def _format_timestamp(self, entry: Dict[str, Any]) -> str:
        ts = entry.get("ts") if isinstance(entry, dict) else None
        if not isinstance(ts, (int, float)) and isinstance(entry, dict):
            payload = entry.get("payload")
            if isinstance(payload, dict):
                ts = payload.get("ts") or payload.get("t")
        if isinstance(ts, (int, float)):
            if ts > 1_000_000_000_000:  # milliseconds
                ts /= 1000.0
            return time.strftime("%H:%M:%S", time.localtime(ts))
        return time.strftime("%H:%M:%S")

    def _summarize_payload(self, payload: Dict[str, Any]) -> str:
        if "payload" in payload and isinstance(payload["payload"], dict):
            return self._summarize_payload(payload["payload"])
        keys = ["name", "ip", "battery", "battery_V", "loop_ms"]
        return codec.summarize_payload(payload, keys=keys, limit=4)

    def _update_status_panel(self, latest: Optional[Dict[str, Any]]) -> None:
        snapshot = self.stream.connection_snapshot()

        payload = self._payload_for_status(latest)

        battery_val = self._extract_float(payload, ["battery", "battery_V", "battery_v"])
        battery_color = self._battery_color(battery_val)
        self._set_status_value("battery", battery_val, color=battery_color)

        ping_ms = snapshot.get("last_latency_ms")
        self._set_status_value("ping", ping_ms)

        heartbeat_age = None
        if snapshot.get("last_message_ts"):
            heartbeat_age = max(0.0, time.time() - snapshot["last_message_ts"])
        elif payload:
            ts = payload.get("ts") or payload.get("t")
            if isinstance(ts, (int, float)):
                if ts > 1_000_000_000_000:
                    ts /= 1000.0
                heartbeat_age = max(0.0, time.time() - ts)
        self._set_status_value("heartbeat", heartbeat_age)

        ip = payload.get("ip") if payload else None
        if not ip:
            ip = self._base_host(snapshot.get("base_url"))
        self._set_status_value("ip", ip)

        opmode = None
        if payload:
            opmode = payload.get("active_opmode") or payload.get("activeOpMode") or payload.get("opmode")
        self._set_status_value("opmode", opmode)

        token = snapshot.get("token")
        if not token and payload:
            token = payload.get("token") or payload.get("token_hint")
        if not token:
            token = self.token_var.get() or None
        self._set_status_value("token", token)

        rate = self._compute_rate()
        self._set_status_value("rate", rate)

        self._set_status_value("dropped", self._dropped_polls)

    def _payload_for_status(self, latest: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not latest:
            return None
        payload = latest.get("payload") if isinstance(latest, dict) else None
        if isinstance(payload, dict):
            combined = dict(payload)
            if "ts" not in combined and isinstance(latest.get("ts"), (int, float)):
                combined["ts"] = latest["ts"]
            for key in ("ip", "token", "token_hint", "active_opmode", "activeOpMode", "opmode"):
                if key in latest and key not in combined and latest[key] is not None:
                    combined[key] = latest[key]
            return combined
        return latest if isinstance(latest, dict) else None

    def _compute_rate(self) -> Optional[float]:
        now = time.time()
        while self._arrival_times and now - self._arrival_times[0] > 5.0:
            self._arrival_times.popleft()
        if not self._arrival_times:
            return None
        span = max(now - self._arrival_times[0], 1e-6)
        return round(len(self._arrival_times) / span, 2)

    def _extract_float(self, payload: Optional[Dict[str, Any]], keys: Iterable[str]) -> Optional[float]:
        if not payload:
            return None
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except (ValueError, TypeError):
                    continue
        return None

    def _battery_color(self, value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        for threshold, color in BATTERY_COLORS:
            if value >= threshold:
                return color
        return BATTERY_COLORS[-1][1]

    def _base_host(self, base_url: Optional[str]) -> Optional[str]:
        if not base_url:
            return None
        if "//" in base_url:
            return base_url.split("//", 1)[1]
        return base_url

    def _set_status_value(self, key: str, value: Any, *, color: Optional[str] = None) -> None:
        self.status_actual[key] = value
        display = self._format_status_value(key, value)
        self.status_vars[key].set(display)
        label = self.status_value_labels.get(key)
        if label and color:
            label.configure(foreground=color)
        elif label:
            label.configure(foreground="#dcdcdc")
        if key == "token":
            self._update_masked_token()

    def _format_status_value(self, key: str, value: Any) -> str:
        if value in (None, ""):
            return "—"
        if key == "battery":
            return f"{value:.2f} V"
        if key == "ping":
            return f"{value:.0f} ms"
        if key == "heartbeat":
            return f"{value:.1f} s"
        if key == "rate":
            return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
        if key == "dropped":
            return str(value)
        if key == "token":
            return str(value)
        return str(value)

    def _toggle_token_visibility(self) -> None:
        self._token_visible = not self._token_visible
        self._update_masked_token()

    def _update_masked_token(self) -> None:
        token = self.status_actual.get("token")
        if token in (None, ""):
            self.status_vars["token"].set("—")
        elif not self._token_visible:
            token_str = str(token)
            masked = "•" * max(4, len(token_str) - 4) + token_str[-4:]
            self.status_vars["token"].set(masked)
        else:
            self.status_vars["token"].set(str(token))

        if not self.token_var.get() and token not in (None, ""):
            token_str = str(token)
            if "*" not in token_str:
                self.token_var.set(token_str)
        if self.token_entry is not None:
            self.token_entry.configure(show="" if self._token_visible else "•")
        self.token_toggle_text.set("Hide" if self._token_visible else "Show")

    # ------------------------------------------------------------------
    # Command sending & history
    def _validate_editor(self) -> None:
        text = self.request_editor.get("1.0", tk.END).strip()
        if not text:
            self.status_bar_var.set("Editor is empty")
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", f"{exc.msg} at line {exc.lineno}")
            return
        try:
            envelope = codec.ensure_command_envelope(data)
        except ValueError as exc:
            messagebox.showerror("Invalid command", str(exc))
            return
        messagebox.showinfo("Valid", codec.pretty_dumps(envelope))

    def _send_request_from_editor(self, _event=None) -> None:
        text = self.request_editor.get("1.0", tk.END).strip()
        if not text:
            self.status_bar_var.set("Editor is empty")
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", f"{exc.msg} at line {exc.lineno}")
            return
        self._dispatch_command(payload, source="editor")

    def _on_library_select(self, _event) -> None:
        selection = self.library_list.curselection()
        if not selection:
            return
        entry = self.saved_commands[selection[0]]
        if entry.get("type") == codec.PING_TYPE:
            envelope = codec.build_envelope(codec.PING_TYPE, entry.get("payload", {}))
        else:
            envelope = codec.build_command(entry["name"], args=entry.get("args"))
        self.request_editor.delete("1.0", tk.END)
        self.request_editor.insert(tk.END, codec.pretty_dumps(envelope))
        self.status_bar_var.set(f"Loaded {entry['label']} into editor")

    def _on_history_double_click(self, _event) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        item_id = selection[0]
        details = self.history_details.get(item_id)
        if not details:
            return
        envelope = details["request"]
        self.request_editor.delete("1.0", tk.END)
        self.request_editor.insert(tk.END, codec.pretty_dumps(envelope))
        self.status_bar_var.set(f"Loaded history item {envelope.get('id', '—')} into editor")

    def _send_console_input(self, _event=None) -> None:
        text = self.console_input.get("1.0", tk.END).strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        self.console_input.delete("1.0", tk.END)
        self._dispatch_command(payload, source="terminal")

    def _send_quick_action(self, name: str) -> None:
        if name == "ping":
            envelope = codec.build_envelope(codec.PING_TYPE, {"source": "quick"})
        elif name == "announce":
            envelope = codec.build_envelope("announce", {"requested": True})
        else:
            envelope = codec.build_command(name)
        self._dispatch_command(envelope, source="quick")

    def _dispatch_command(self, command: Any, *, source: str) -> None:
        if isinstance(command, str):
            envelope = command
        else:
            try:
                envelope = codec.ensure_command_envelope(command)
            except ValueError as exc:
                messagebox.showerror("Invalid command", str(exc))
                return
        self._append_output(f"[{source}] → {self._safe_dump(envelope)}")
        response = self.stream.send_command(envelope)
        self._append_output(f"[{source}] ← {self._safe_dump(response)}")
        self._record_history(envelope, response)

    def _record_history(self, envelope: Any, response: Dict[str, Any]) -> None:
        if isinstance(envelope, str):
            request_id = "raw"
            name = envelope[:40]
        else:
            request_id = envelope.get("id", "—")
            name = envelope.get("payload", {}).get("name", envelope.get("type", "command"))
        if isinstance(response, dict):
            status = "OK" if response.get("ok", True) else "ERR"
        else:
            status = "OK"
        timestamp = time.strftime("%H:%M:%S")
        item_id = self.history_tree.insert("", tk.END, values=(request_id, name, status, timestamp))
        self.history_details[item_id] = {"request": envelope, "response": response}
        self.command_history.append({"request": envelope, "response": response, "ts": timestamp})

    def _append_output(self, text: str) -> None:
        self.console_output.configure(state=tk.NORMAL)
        self.console_output.insert(tk.END, text + "\n")
        self.console_output.configure(state=tk.DISABLED)
        self.console_output.see(tk.END)

    def _safe_dump(self, obj: Any) -> str:
        try:
            if isinstance(obj, str):
                return obj
            return codec.pretty_dumps(obj)
        except Exception:
            return str(obj)


__all__ = ["DevControllerApp"]
