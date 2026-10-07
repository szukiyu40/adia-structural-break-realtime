"""Check source bytes and syntax without importing or running any model."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "source_manifest.json").read_text())
python_files = 0
changed_files = 0
patterns = {
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "api_token": r"\b(?:sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})",
    "signed_url": r"[?&](?:token|access_token|api_key|signature|X-Amz-Signature|X-Goog-Signature)=",
}
for relative, record in manifest["files"].items():
    path = ROOT / relative
    assert path.is_file(), relative
    raw = path.read_bytes()
    assert len(raw) == record["bytes"], relative
    assert hashlib.sha256(raw).hexdigest() == record["sha256"], relative
    source = raw.decode("utf-8")
    assert "/Users/" not in source, relative
    assert "drive.google.com" not in source, relative
    if path.suffix == ".py":
        ast.parse(source, filename=relative)
        python_files += 1
    for label, pattern in patterns.items():
        assert re.search(pattern, source) is None, (relative, label)
    changed_files += record["publication_change"] != "none"
assert python_files == 174, python_files
actual = {str(p.relative_to(ROOT)) for p in (ROOT / "submission").rglob("*") if p.is_file()}
assert actual == set(manifest["files"]), "Unexpected submission files"
print(f"PASS: {python_files} Python sources; {changed_files} files with documented publication changes; source SHA256 and syntax verified. No model code was executed.")
