from __future__ import annotations

import argparse
from pathlib import Path

from wesi.application.services import WesiService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wesi MVP command line")
    parser.add_argument("command", choices=["ui", "init-project", "demo-run"], help="Action to execute")
    parser.add_argument("--root", default=None, help="Project workspace root")
    parser.add_argument("--name", default="Wesi Project", help="Project name for init-project")
    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else Path.cwd() / "artifacts" / "wesi-project"
    service = WesiService(root)

    if args.command == "init-project":
        service.create_project(args.name)
        print(f"Initialized project at {root}")
        return 0

    if args.command == "demo-run":
        from wesi.ui.main_window import WesiMainWindow

        WesiMainWindow(service).build_demo()
        job_id = service.create_job(max_workers=1)
        result = service.run_job(job_id, backend="numpy")
        print(result)
        return 0

    from wesi.ui.main_window import launch

    return launch(service)


if __name__ == "__main__":
    raise SystemExit(main())
