"""Command-line entry point for nlos_cs.

This is intentionally minimal at the start of the rebuild.
It provides:
- package version
- a stable console entry point
- placeholder subcommands for later experiment wiring
"""

from __future__ import annotations

import argparse
from typing import Sequence

from nlos_cs import __version__


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

    # Version/info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show package and CLI information",
    )
    info_parser.set_defaults(handler=_handle_info)

    # Placeholder for future experiment wiring
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
    print("Available commands: info, build-operator, reconstruct, discrim, robustness")
    return 0


def _handle_placeholder(args: argparse.Namespace) -> int:
    """Handle placeholder commands until config-driven CLI wiring is added."""
    command = getattr(args, "command", None) or "unknown"
    print(
        f"Command '{command}' is not wired yet. "
        "Use the Python experiment modules directly for now."
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