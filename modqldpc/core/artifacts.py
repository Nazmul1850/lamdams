from __future__ import annotations
import json, os, shutil
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone

def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj

class ArtifactStore:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

    @staticmethod
    def make_run_dir(base: str = "runs", *, tag: str = "run") -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_id = f"{tag}__{ts}"
        run_dir = os.path.join(base, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def put_text(self, relpath: str, text: str) -> str:
        abspath = os.path.join(self.run_dir, relpath)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", encoding="utf-8") as f:
            f.write(text)
        return abspath

    def put_json(self, relpath: str, obj: Any) -> str:
        abspath = os.path.join(self.run_dir, relpath)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        with open(abspath, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(obj), f, indent=2, sort_keys=True)
        return abspath

    def copy_in(self, src_path: str, relpath: str) -> str:
        abspath = os.path.join(self.run_dir, relpath)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        shutil.copyfile(src_path, abspath)
        return abspath
