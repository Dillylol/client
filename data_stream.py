
import json
import requests
from typing import List, Optional
from metrics_datamodel import Metrics

class DataStream:
    def __init__(self,
                 base_url: str = "http://192.168.43.1:8080",
                 dump_path: str = "/jules/dump",
                 cmd_path: str = "/jules/cmd"):
        self.base_url = base_url.rstrip("/")
        self.dump_url = f"{self.base_url}{dump_path}"
        self.cmd_url = f"{self.base_url}{cmd_path}"
        self.session = requests.Session()

    def get_data(self) -> Optional[List[Metrics]]:
        try:
            resp = self.session.get(self.dump_url, timeout=1)
            resp.raise_for_status()
            data = resp.json()

            metrics_list: List[Metrics] = []
            for item in data:
                # Seed a Metrics instance with any known/annotated fields
                allowed = Metrics.__annotations__.keys()
                m = Metrics(**{k: item.get(k) for k in allowed if k in item})

                # Merge any additional top-level keys dynamically
                for k, v in item.items():
                    if k not in allowed:
                        m.set_dynamic(k, v)

                # If there is nested jsonData, parse and merge
                if item.get("jsonData"):
                    try:
                        nested = json.loads(item["jsonData"])
                        for k, v in nested.items():
                            m.set_dynamic(k, v)
                    except json.JSONDecodeError:
                        # Keep going even if nested JSON is malformed
                        pass

                metrics_list.append(m)
            return metrics_list

        except requests.exceptions.RequestException:
            return None
        except json.JSONDecodeError:
            return None

    def send_command(self, command: str) -> dict:
        """
        Send a terminal/console command to the robot.
        Tries POST JSON first; falls back to GET with query param 'q'.
        Returns a response dict, or {'ok': False, 'error': '...'} on failure.
        """
        try:
            r = self.session.post(self.cmd_url, json={"cmd": command}, timeout=2)
            if r.ok:
                try:
                    return r.json()
                except Exception:
                    return {"ok": True, "text": r.text}
        except requests.exceptions.RequestException as e:
            # Fall through to GET
            pass

        try:
            r = self.session.get(self.cmd_url, params={"q": command}, timeout=2)
            if r.ok:
                try:
                    return r.json()
                except Exception:
                    return {"ok": True, "text": r.text}
            else:
                return {"ok": False, "status": r.status_code, "text": r.text}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": str(e)}
