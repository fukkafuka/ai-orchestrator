from pathlib import Path

ROOT = Path.home() / "ai-orchestrator"

def read_file(path: str, start: int = 1, lines: int = 200) -> str:
    target = (ROOT / path).resolve()

    if not str(target).startswith(str(ROOT.resolve())):
        raise PermissionError("Access denied")

    if not target.exists():
        return f"{path} not found"

    data = target.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines()

    start = max(start, 1)
    end = min(start + lines - 1, len(data))

    result = []

    for no in range(start, end + 1):
        result.append(f"{no:5}: {data[no-1]}")

    return "\n".join(result)

if __name__ == "__main__":
    import sys

    filename = "orchestrator_v4.py"
    start = 1
    lines = 30

    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    if len(sys.argv) >= 3:
        start = int(sys.argv[2])

    if len(sys.argv) >= 4:
        lines = int(sys.argv[3])

    print(read_file(filename, start, lines))
