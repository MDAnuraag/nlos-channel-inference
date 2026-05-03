"""Command-line entry point for nlos_cs.

This version keeps the CLI small, but now supports config-driven experiment
execution through JSON config files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from nlos_cs import __version__
from nlos_cs.cli.run_experiment import run_experiment_from_config
from nlos_cs.io.config import load_json_config


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="nlos-cs",
        description=(
            "Metasurface-assisted NLOS sensing and imaging with an explicit "
            "compressed sensing and inverse-problems viewpoint."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    info_parser = subparsers.add_parser(
        "info",
        help="Show package and CLI information",
    )
    info_parser.set_defaults(handler=_handle_info)

    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from a JSON config file",
    )
    run_parser.add_argument(
        "config",
        help="Path to a JSON config file",
    )
    run_parser.set_defaults(handler=_handle_run)

    build_parser_cmd = subparsers.add_parser(
        "build-operator",
        help="Placeholder for operator-build experiment CLI",
    )
    build_parser_cmd.set_defaults(handler=_handle_placeholder)

    recon_parser_cmd = subparsers.add_parser(
        "reconstruct",
        help="Placeholder for reconstruction experiment CLI",
    )
    recon_parser_cmd.set_defaults(handler=_handle_placeholder)

    disc_parser_cmd = subparsers.add_parser(
        "discrim",
        help="Placeholder for discrimination experiment CLI",
    )
    disc_parser_cmd.set_defaults(handler=_handle_placeholder)

    robust_parser_cmd = subparsers.add_parser(
        "robustness",
        help="Placeholder for robustness experiment CLI",
    )
    robust_parser_cmd.set_defaults(handler=_handle_placeholder)

    return parser


def _handle_info(_: argparse.Namespace) -> int:
    """Handle the 'info' subcommand."""
    print(f"nlos-cs version: {__version__}")
    print("Status: early rebuild")
    print("Available commands: info, run, build-operator, reconstruct, discrim, robustness")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    """Handle the 'run' subcommand using a JSON config file."""
    config_path = Path(args.config)
    loaded = load_json_config(config_path)
    result = run_experiment_from_config(loaded.data)

    print(f"experiment_type: {result['experiment_type']}")
    if "solver_name" in result:
        print(f"solver_name: {result['solver_name']}")
    if "run_dir" in result:
        print(f"run_dir: {result['run_dir']}")

    for key, value in result.items():
        if key in {"experiment_type", "solver_name", "run_dir"}:
            continue
        print(f"{key}: {value}")

    return 0


def _handle_placeholder(args: argparse.Namespace) -> int:
    """Handle placeholder commands until direct CLI wiring is added."""
    command = getattr(args, "command", None) or "unknown"
    print(
        f"Command '{command}' is not wired yet. "
        "Use 'run <config.json>' or the Python experiment modules directly for now."
    )
    return 0


def app(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point used by pyproject.toml."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(app())