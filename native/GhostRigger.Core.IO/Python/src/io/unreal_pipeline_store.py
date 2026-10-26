"""Durable build receipts and recoverable, conflict-checked retail file writes."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path):
    return digest(path.read_bytes()) if path.is_file() else None


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".ghost-write-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(path: Path, value: dict):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def read_receipt(path: Path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Job receipt exceeds its size limit.")
    return json.loads(path.read_text(encoding="utf-8"))


def _target(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if len(parts) != 2 or parts[0] not in {"Modules", "Override"} or any(c in relative for c in "\\:") or parts[1] in {".", ".."}:
        raise ValueError("Install targets must be direct Modules/Override resource files.")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Install target escapes the selected game folder.")
    return path


@contextmanager
def _install_lock(root: Path):
    """OS-owned lock releases on process exit; lock file is not a stale flag."""
    import msvcrt
    with (root / ".ghoststudio-install.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another Ghost install or rollback is active.") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _restore(receipt_path: Path, receipt: dict, require_game_closed):
    root = Path(receipt["game_root"])
    failures = []
    for row in reversed(receipt["files"]):
        try:
            require_game_closed()
            target = _target(root, row["target"])
            current = file_digest(target)
            if current == row["previous_sha256"]:
                row["state"] = "restored"
                continue
            if current != row["installed_sha256"]:
                raise ValueError("Destination changed outside this transaction; preserving it and its backup.")
            if row["backup"]:
                backup = receipt_path.parent / row["backup"]
                if not backup.resolve().is_relative_to(receipt_path.parent.resolve()):
                    raise ValueError("Recovery backup escapes its transaction folder.")
                data = backup.read_bytes()
                if digest(data) != row["previous_sha256"]:
                    raise ValueError("Recovery backup hash does not match.")
                atomic_bytes(target, data)
            else:
                target.unlink()  # Only this exact installed file; original was absent.
            row["state"] = "restored"
            write_receipt(receipt_path, receipt)
        except Exception as error:
            failures.append(f"{row['target']}: {error}")
    receipt.update(state="recovery_required" if failures else "rolled_back", recovery_errors=failures)
    write_receipt(receipt_path, receipt)
    return receipt


def install_resources(game_root, transaction_dir, files, *, require_game_closed):
    """Files are (relative destination, bytes). Never overwrite source or saves."""
    root, transaction = Path(game_root).resolve(), Path(transaction_dir).resolve()
    if not root.is_dir() or transaction == root or transaction.is_relative_to(root):
        raise ValueError("Recovery storage must be outside an existing game folder.")
    if not files or len({name.casefold() for name, _ in files}) != len(files):
        raise ValueError("An install requires distinct resource destinations.")
    for name, _ in files:
        _target(root, name)
    transaction.mkdir(parents=True, exist_ok=False)
    receipt_path = transaction / "installation.json"
    with _install_lock(root):
        require_game_closed()
        receipt = {"schema": "ghoststudio.retail-transaction/v1", "game_root": str(root), "state": "preparing", "files": []}
        for index, (name, data) in enumerate(files):
            target = _target(root, name)
            old = target.read_bytes() if target.exists() else None
            backup = f"backups/{index:04d}-{digest(name.encode())}.bin" if old is not None else ""
            if old is not None:
                atomic_bytes(transaction / backup, old)
            receipt["files"].append({"target": name, "previous_sha256": digest(old) if old is not None else None,
                "installed_sha256": digest(data), "backup": backup, "state": "prepared"})
        receipt["state"] = "installing"
        write_receipt(receipt_path, receipt)
        try:
            for row, (_, data) in zip(receipt["files"], files):
                require_game_closed()
                target = _target(root, row["target"])
                if file_digest(target) != row["previous_sha256"]:
                    raise ValueError(f"Destination changed after install preview: {row['target']}")
                atomic_bytes(target, data)
                if file_digest(target) != row["installed_sha256"]:
                    raise ValueError(f"Installed bytes did not verify: {row['target']}")
                row["state"] = "installed"
                write_receipt(receipt_path, receipt)
            receipt["state"] = "installed"
            write_receipt(receipt_path, receipt)
        except Exception as error:
            receipt["failure"] = str(error)
            _restore(receipt_path, receipt, require_game_closed)
            raise RuntimeError(f"Install failed: {error}. Recovery receipt: {receipt_path}") from error
    return receipt_path


def rollback_install(receipt_path, *, require_game_closed):
    path = Path(receipt_path).resolve()
    receipt = read_receipt(path)
    if receipt.get("schema") != "ghoststudio.retail-transaction/v1":
        raise ValueError("Unsupported retail recovery receipt.")
    with _install_lock(Path(receipt["game_root"])):
        return _restore(path, receipt, require_game_closed)
