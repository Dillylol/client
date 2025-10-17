from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Callable

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

import codec
from ui_bridge.client_api import ClientAPI

FILTER_TAGS = [
    ("heartbeat", "Heartbeat"),
    ("snapshot", "Snapshots"),
    ("diff", "Diffs"),
]


class DevControllerApp(tk.Tk):
    """Dashboard for the Jules FTC telemetry client."""

    def __init__(self, client: Optional[ClientAPI] = None) -> None:
        super().__init__()
        self.title("JULES Dev Controller")
        self.geometry("1360x840")
        self.configure(bg="#1f1f1f")

        self.api = client or ClientAPI()

        self.status_vars: Dict[str, tk.StringVar] = {}
        self.status_actual: Dict[str, Any] = {}
        self.status_masked_keys: set[str] = {"token"}

        self.connection_state_var = tk.StringVar(value="DISCONNECTED")
        self.status_bar_var = tk.StringVar(value="Ready")
        self.base_url_var = tk.StringVar(value="ws://")
        self.token_var = tk.StringVar(value="")
        self.token_visible = False

        self.metrics_snapshot: Dict[str, Any] = {}
        self.flat_map: Dict[str, Any] = {}
        self.tree_data: Dict[str, Any] = {}
        self._tree_paths: Dict[str, str] = {}
        self._search_results: List[str] = []
        self._log_filters: Dict[str, tk.BooleanVar] = {}

        self.saved_commands: List[Dict[str, Any]] = [
            {"label": "Drive Forward", "name": "drive", "args": {"t": 0.5, "p": 0.5, "duration_ms": 400}},
            {"label": "Strafe Right", "name": "strafe", "args": {"speed": 0.4, "duration_ms": 400}},
            {"label": "Turn Left", "name": "turn", "args": {"speed": -0.4, "degrees": 30}},
            {"label": "Stop", "name": "stop", "args": {}},
            {"label": "Ping", "type": codec.PING_TYPE, "payload": {"source": "client"}},
        ]

        self._build_ui()
        self._wire_callbacks()

    # ------------------------------------------------------------------
    # UI construction
    def _wire_callbacks(self) -> None:
        self.api.on_status = lambda status: self.after(0, self._handle_status, status)
        self.api.on_metrics = lambda metrics: self.after(0, self._handle_metrics, metrics)
        self.api.on_telemetry_tree = lambda tree: self.after(0, self._update_tree_view, tree)
        self.api.on_flat = lambda flat: self.after(0, self._handle_flat_map, flat)
        self.api.on_stdout = lambda line: self.after(0, self._append_stdout, line)
        self.api.on_raw_frame = lambda frame: self.after(0, self._append_log_entry, frame)

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
        style.configure("Dark.TEntry", fieldbackground="#262626", foreground="#f5f5f5", insertcolor="#f5f5f5")
        style.configure("Treeview", background="#262626", fieldbackground="#262626", foreground="#e0e0e0")
        style.configure("Treeview.Heading", background="#333333", foreground="#f0f0f0")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("App.TCheckbutton", background="#1f1f1f", foreground="#f5f5f5")
        style.map("App.TCheckbutton", background=[("selected", "#1f1f1f"), ("!selected", "#1f1f1f")], foreground=[("selected", "#9cdcfe"), ("!selected", "#f5f5f5")])

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_pane.grid(row=0, column=0, sticky="nsew")

        status_panel = self._build_status_panel(main_pane)
        main_pane.add(status_panel, weight=0)

        right_pane = ttk.Panedwindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane, weight=1)

        notebook_frame = ttk.Frame(right_pane, padding=12, style="App.TFrame")
        right_pane.add(notebook_frame, weight=3)

        self.nb = ttk.Notebook(notebook_frame)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.telemetry_tab = ttk.Frame(self.nb, padding=12, style="App.TFrame")
        self.nb.add(self.telemetry_tab, text="Telemetry")
        self._build_telemetry_tab(self.telemetry_tab)

        self.requests_tab = ttk.Frame(self.nb, padding=12, style="App.TFrame")
        self.nb.add(self.requests_tab, text="Commands")
        self._build_requests_tab(self.requests_tab)

        console_pane = ttk.Panedwindow(right_pane, orient=tk.HORIZONTAL)
        right_pane.add(console_pane, weight=2)
        self._build_console(console_pane)

        status_bar = ttk.Label(self, textvariable=self.status_bar_var, anchor="w", style="App.TLabel")
        status_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

    def _build_status_panel(self, parent: ttk.Panedwindow) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=16, style="App.TFrame")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Connection", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        conn_row = ttk.Frame(frame, style="App.TFrame")
        conn_row.grid(row=1, column=0, sticky="ew", pady=(6, 12))

        self.status_dot = tk.Canvas(conn_row, width=14, height=14, highlightthickness=0, bg="#1f1f1f")
        self.status_dot.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_status_dot("#7f7f7f")

        ttk.Label(conn_row, textvariable=self.connection_state_var, style="StatusValue.TLabel").pack(side=tk.LEFT)

        ttk.Label(frame, text="Robot Endpoint", style="StatusHeader.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.base_url_var, style="Dark.TEntry").grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(frame, text="Token", style="StatusHeader.TLabel").grid(row=4, column=0, sticky="w")
        token_row = ttk.Frame(frame, style="App.TFrame")
        token_row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.token_entry = ttk.Entry(token_row, textvariable=self.token_var, style="Dark.TEntry", show="•")
        self.token_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(token_row, text="Show", command=self._toggle_token_visibility).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(token_row, text="Copy", command=lambda: self._copy_to_clipboard(self.token_var.get())).pack(side=tk.LEFT, padx=(8, 0))

        btn_row = ttk.Frame(frame, style="App.TFrame")
        btn_row.grid(row=6, column=0, sticky="ew", pady=(0, 16))
        ttk.Button(btn_row, text="Connect", command=self._on_connect, style="Accent.TButton").pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Disconnect", command=self._on_disconnect).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_row, text="Ping", command=self._manual_ping).pack(side=tk.LEFT, padx=(8, 0))

        row = 7
        for key, label in [
            ("battery", "Battery"),
            ("ping", "Ping"),
            ("heartbeat", "Heartbeat Age"),
            ("opmode", "Active OpMode"),
        ]:
            self._create_status_row(frame, row, key, label)
            row += 2

        return frame

    def _create_status_row(self, frame: ttk.Frame, row: int, key: str, label: str) -> None:
        ttk.Label(frame, text=label, style="StatusHeader.TLabel").grid(row=row, column=0, sticky="w")
        value_var = tk.StringVar(value="—")
        ttk.Label(frame, textvariable=value_var, style="StatusValue.TLabel").grid(row=row + 1, column=0, sticky="w", pady=(0, 6))
        self.status_vars[key] = value_var
        self.status_actual[key] = None

    def _build_telemetry_tab(self, frame: ttk.Frame) -> None:
        search_row = ttk.Frame(frame, style="App.TFrame")
        search_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(search_row, text="Search", style="StatusHeader.TLabel").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        entry = ttk.Entry(search_row, textvariable=self.search_var, style="Dark.TEntry")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        entry.bind("<KeyRelease>", lambda _e: self._update_search_results())
        ttk.Button(search_row, text="Copy Path", command=self._copy_selected_path).pack(side=tk.LEFT)
        ttk.Button(search_row, text="Export CSV", command=self._export_csv).pack(side=tk.LEFT, padx=(8, 0))

        results_frame = ttk.Frame(frame, style="App.TFrame")
        results_frame.pack(fill=tk.X, pady=(0, 12))
        self.search_results = tk.Listbox(results_frame, height=6, background="#1e1e1e", foreground="#d4d4d4")
        self.search_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.search_results.bind("<Double-1>", lambda _e: self._copy_selected_path())
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_results.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.search_results.configure(yscrollcommand=scrollbar.set)

        tree_frame = ttk.Frame(frame, style="App.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree_view = ttk.Treeview(tree_frame, columns=("value",), show="tree headings")
        self.tree_view.heading("#0", text="Path", anchor="w")
        self.tree_view.heading("value", text="Value", anchor="w")
        self.tree_view.column("#0", width=260)
        self.tree_view.column("value", width=200)
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree_view.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_view.configure(yscrollcommand=tree_scroll.set)
        self.tree_view.pack(fill=tk.BOTH, expand=True)
        self.tree_view.bind("<Button-3>", self._on_tree_right_click)
        self.tree_view.bind("<Double-1>", self._copy_tree_selection)

        log_frame = ttk.Frame(frame, style="App.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        ttk.Label(log_frame, text="Frame Log", style="Title.TLabel").pack(anchor="w", pady=(0, 6))
        filters_row = ttk.Frame(log_frame, style="App.TFrame")
        filters_row.pack(anchor="w", pady=(0, 6))
        for tag, label in FILTER_TAGS:
            var = tk.BooleanVar(value=True)
            self._log_filters[tag] = var
            ttk.Checkbutton(filters_row, text=label, variable=var, style="App.TCheckbutton", command=self._refresh_log_display).pack(side=tk.LEFT, padx=(0, 8))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state=tk.DISABLED, background="#252526", foreground="#d4d4d4")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_entries: List[Dict[str, Any]] = []

    def _build_requests_tab(self, frame: ttk.Frame) -> None:
        pane = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        library_frame = ttk.Frame(pane, padding=(0, 0, 12, 0), style="App.TFrame")
        pane.add(library_frame, weight=1)
        ttk.Label(library_frame, text="Saved Commands", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.library_list = tk.Listbox(library_frame, height=10, background="#1e1e1e", foreground="#d4d4d4", selectbackground="#264f78")
        self.library_list.pack(fill=tk.BOTH, expand=True)
        for entry in self.saved_commands:
            self.library_list.insert(tk.END, entry["label"])
        self.library_list.bind("<<ListboxSelect>>", self._on_library_select)

        editor_frame = ttk.Frame(pane, style="App.TFrame")
        pane.add(editor_frame, weight=3)
        ttk.Label(editor_frame, text="Command Editor", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.request_editor = scrolledtext.ScrolledText(editor_frame, height=12, background="#1e1e1e", foreground="#d4d4d4")
        self.request_editor.pack(fill=tk.BOTH, expand=True)
        self.request_editor.bind("<Control-Return>", lambda _e: self._send_request())

        button_row = ttk.Frame(editor_frame, style="App.TFrame")
        button_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(button_row, text="Validate", command=self._validate_editor).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Send", command=self._send_request).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Clear", command=lambda: self.request_editor.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(8, 0))

        history_frame = ttk.Frame(editor_frame, padding=(0, 12, 0, 0), style="App.TFrame")
        history_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(history_frame, text="History", style="Title.TLabel").pack(anchor="w", pady=(0, 6))
        columns = ("ts", "payload")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=6)
        self.history_tree.heading("ts", text="Time")
        self.history_tree.heading("payload", text="Payload")
        self.history_tree.column("ts", width=140)
        self.history_tree.column("payload", width=320)
        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

    def _build_console(self, pane: ttk.Panedwindow) -> None:
        input_frame = ttk.Frame(pane, padding=12, style="App.TFrame")
        pane.add(input_frame, weight=1)
        ttk.Label(input_frame, text="Input Terminal", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.console_input = scrolledtext.ScrolledText(input_frame, height=12, wrap=tk.NONE, background="#1e1e1e", foreground="#d4d4d4")
        self.console_input.pack(fill=tk.BOTH, expand=True)
        self.console_input.bind("<Control-Return>", lambda _e: self._send_console_input())

        console_btns = ttk.Frame(input_frame, style="App.TFrame")
        console_btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(console_btns, text="Send", command=self._send_console_input).pack(side=tk.LEFT)
        ttk.Button(console_btns, text="Clear", command=lambda: self.console_input.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(8, 0))

        output_frame = ttk.Frame(pane, padding=12, style="App.TFrame")
        pane.add(output_frame, weight=1)
        ttk.Label(output_frame, text="Output Terminal", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.console_output = scrolledtext.ScrolledText(output_frame, height=12, wrap=tk.WORD, state=tk.DISABLED, background="#1e1e1e", foreground="#c7c7c7")
        self.console_output.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Connection handlers
    def _on_connect(self) -> None:
        url = self.base_url_var.get().strip()
        if not url:
            messagebox.showerror("Invalid URL", "Please enter the robot endpoint URL")
            return
        token = self.token_var.get().strip() or None
        try:
            self.api.configure(url, token)
            self.api.start()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Connection Error", str(exc))
            return
        self.status_bar_var.set(f"Connecting to {url}")
        if token:
            self.status_actual["token"] = token
        else:
            self.status_actual.pop("token", None)
        self.connection_state_var.set("CONNECTING…")

    def _on_disconnect(self) -> None:
        self.api.stop()
        self.connection_state_var.set("DISCONNECTED")
        self._draw_status_dot("#7f7f7f")
        self.status_bar_var.set("Disconnected")

    def _manual_ping(self) -> None:
        self.api.manual_ping()
        self.status_bar_var.set("Ping sent")

    # ------------------------------------------------------------------
    # API callbacks
    def _handle_status(self, status: Dict[str, Any]) -> None:
        state = str(status.get("state", "")).upper()
        if state:
            self.connection_state_var.set(state)
        message = status.get("last_error") or status.get("endpoint") or status.get("state")
        if message:
            self.status_bar_var.set(str(message))

    def _handle_metrics(self, metrics: Dict[str, Any]) -> None:
        self.metrics_snapshot = metrics
        battery = metrics.get("battery_v")
        ping = metrics.get("ping_ms")
        heartbeat = metrics.get("heartbeat_age_ms")
        opmode = metrics.get("active_opmode")
        connected = bool(metrics.get("connected"))

        self._set_status_value("battery", battery, formatter=lambda v: f"{v:.2f} V")
        self._set_status_value("ping", ping, formatter=lambda v: f"{v:.0f} ms")
        self._set_status_value("heartbeat", heartbeat, formatter=lambda v: f"{v:.0f} ms")
        self._set_status_value("opmode", opmode)
        self._draw_status_dot("#2ecc71" if connected else "#ef5350")

    def _handle_flat_map(self, flat: Dict[str, Any]) -> None:
        self.flat_map = flat
        self._update_search_results()

    def _update_tree_view(self, tree: Dict[str, Any]) -> None:
        self.tree_data = tree
        self.tree_view.delete(*self.tree_view.get_children())
        self._tree_paths.clear()
        for key in sorted(tree.keys()):
            node_id = self._insert_tree_node("", key, tree[key], key)
            self.tree_view.item(node_id, open=True)

    def _append_stdout(self, line: str) -> None:
        self.console_output.configure(state=tk.NORMAL)
        self.console_output.insert(tk.END, line + "\n")
        self.console_output.configure(state=tk.DISABLED)
        self.console_output.see(tk.END)

    def _append_log_entry(self, frame: Dict[str, Any]) -> None:
        entry = {
            "type": frame.get("type", "unknown"),
            "ts_ms": frame.get("ts_ms") or frame.get("ts") or int(time.time() * 1000),
            "frame": frame,
        }
        self.log_entries.append(entry)
        self.log_entries = self.log_entries[-500:]
        self._refresh_log_display()

    # ------------------------------------------------------------------
    # Utility helpers
    def _set_status_value(self, key: str, value: Any, *, formatter: Optional[Callable[[Any], str]] = None) -> None:
        self.status_actual[key] = value
        if value in (None, ""):
            display = "—"
        elif formatter:
            try:
                display = formatter(value)
            except Exception:  # noqa: BLE001
                display = str(value)
        else:
            display = str(value)
        var = self.status_vars.get(key)
        if var:
            var.set(display)

    def _draw_status_dot(self, color: str) -> None:
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

    def _copy_to_clipboard(self, text: str) -> None:
        if not text:
            self.status_bar_var.set("Nothing to copy")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_bar_var.set("Copied to clipboard")

    def _toggle_token_visibility(self) -> None:
        self.token_visible = not self.token_visible
        self.token_entry.configure(show="" if self.token_visible else "•")

    def _insert_tree_node(self, parent: str, key: str, value: Any, path: str) -> str:
        if isinstance(value, dict):
            node_id = self.tree_view.insert(parent, "end", text=key, values=("",))
            self._tree_paths[node_id] = path
            for child_key in sorted(value.keys()):
                child_path = f"{path}.{child_key}" if path else child_key
                self._insert_tree_node(node_id, child_key, value[child_key], child_path)
        elif isinstance(value, list):
            node_id = self.tree_view.insert(parent, "end", text=f"{key} [list]", values=(f"{len(value)} items",))
            self._tree_paths[node_id] = path
            for idx, item in enumerate(value):
                child_path = f"{path}.{idx}" if path else str(idx)
                self._insert_tree_node(node_id, f"[{idx}]", item, child_path)
        else:
            display = value if isinstance(value, (str, int, float, bool)) else repr(value)
            node_id = self.tree_view.insert(parent, "end", text=key, values=(display,))
            self._tree_paths[node_id] = path
        return node_id

    def _update_search_results(self) -> None:
        query = self.search_var.get().strip().lower()
        self.search_results.delete(0, tk.END)
        self._search_results.clear()
        if not self.flat_map:
            return
        for key in sorted(self.flat_map.keys()):
            if query and query not in key.lower():
                continue
            value = self.flat_map[key]
            display = value if isinstance(value, (str, int, float, bool)) else repr(value)
            line = f"{key} = {display}"
            self.search_results.insert(tk.END, line)
            self._search_results.append(key)

    def _copy_selected_path(self) -> None:
        selection = self.search_results.curselection()
        if not selection:
            return
        idx = selection[0]
        key = self._search_results[idx]
        self._copy_to_clipboard(key)

    def _on_tree_right_click(self, event: tk.Event) -> None:  # type: ignore[name-defined]
        item = self.tree_view.identify_row(event.y)
        if item:
            self.tree_view.selection_set(item)
            self._copy_tree_selection()

    def _copy_tree_selection(self, *_args: Any) -> None:
        selection = self.tree_view.selection()
        if not selection:
            return
        path = self._tree_paths.get(selection[0])
        if path:
            self._copy_to_clipboard(path)

    def _refresh_log_display(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for entry in self.log_entries:
            frame_type = entry.get("type", "unknown")
            filter_var = self._log_filters.get(frame_type)
            if filter_var is not None and not filter_var.get():
                continue
            ts_ms = entry.get("ts_ms")
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000)) if isinstance(ts_ms, (int, float)) else "--"
            self.log_text.insert(tk.END, f"[{ts_str}] {frame_type}: {entry['frame']}\n")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _export_csv(self) -> None:
        if not self.flat_map:
            messagebox.showinfo("Export", "No telemetry data available yet.")
            return
        selection = self.search_results.curselection()
        if selection:
            keys = [self._search_results[idx] for idx in selection]
        else:
            keys = list(self.flat_map.keys())
        if not keys:
            messagebox.showinfo("Export", "No keys selected for export.")
            return
        path = filedialog.asksaveasfilename(title="Export CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            data = self.api.ingester.export_csv(keys, window_ms=5 * 60 * 1000)
            with open(path, "wb") as fh:
                fh.write(data)
            self.status_bar_var.set(f"Exported CSV to {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export Failed", str(exc))

    # ------------------------------------------------------------------
    # Command handling
    def _on_library_select(self, _event: tk.Event) -> None:  # type: ignore[name-defined]
        selection = self.library_list.curselection()
        if not selection:
            return
        entry = self.saved_commands[selection[0]]
        self.request_editor.delete("1.0", tk.END)
        self.request_editor.insert(tk.END, json.dumps(entry, indent=2))

    def _validate_editor(self) -> None:
        text = self.request_editor.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Validate", "Editor is empty")
            return
        try:
            json.loads(text)
            messagebox.showinfo("Validate", "Valid JSON payload")
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", str(exc))

    def _send_request(self) -> None:
        text = self.request_editor.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Send", "Nothing to send")
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", str(exc))
            return
        self.api.send_cmd(json.dumps(payload))
        ts = time.strftime("%H:%M:%S")
        self.history_tree.insert("", tk.END, values=(ts, json.dumps(payload)))
        self.status_bar_var.set("Command sent")

    def _send_console_input(self) -> None:
        text = self.console_input.get("1.0", tk.END).strip()
        if not text:
            return
        self.api.send_cmd(text)
        self.console_input.delete("1.0", tk.END)
        self.status_bar_var.set("Console command sent")


__all__ = ["DevControllerApp"]
