#!/usr/bin/env python3
"""
version.py

Safely report versions of commonly used packages in this project.

Usage:
  python version.py                 # print human-readable versions
  python version.py --json out.json # write JSON to out.json
  python version.py --freeze        # also run `pip freeze` and include output

The script is defensive: it marks packages as 'not installed' when unavailable
and tries multiple ways of discovering a package's version.
"""

from __future__ import annotations

import sys
import argparse
import json
import subprocess
import importlib
from importlib import import_module

try:
	# Python 3.8+
	from importlib.metadata import version as _get_dist_version, PackageNotFoundError
except Exception:
	try:
		# pkg_resources as fallback
		from pkg_resources import get_distribution as _get_dist_version  # type: ignore
		from pkg_resources import DistributionNotFound as PackageNotFoundError  # type: ignore
	except Exception:
		_get_dist_version = None
		PackageNotFoundError = Exception


COMMON_PACKAGES = [
	"python",
	"numpy",
	"scipy",
	"matplotlib",
	"trimesh",
	"meshpy",
	"meshio",
	"dolfinx",
	"petsc4py",
	"mpi4py",
	"ufl",
	"basix",
]


def get_version_by_import(pkg_name: str) -> str:
	"""Try to import pkg_name and obtain a version string from common attributes."""
	if pkg_name == "python":
		return sys.version.split()[0]

	try:
		mod = import_module(pkg_name)
	except Exception:
		return "not installed"

	# Common attributes for version strings
	for attr in ("__version__", "version", "__VERSION__"):
		v = getattr(mod, attr, None)
		if isinstance(v, str):
			return v

	# Some packages don't expose __version__; try metadata-based lookup
	if _get_dist_version:
		try:
			v = _get_dist_version(pkg_name)
			# pkg_resources.get_distribution returns a Distribution object; prefer .version
			return getattr(v, "version", str(v))
		except PackageNotFoundError:
			# Try common alternative names
			alt_names = {
				"trimesh": "trimesh",
				"meshio": "meshio",
				"meshpy": "meshpy",
				"petsc4py": "petsc4py",
				"dolfinx": "dolfinx",
			}
			alt = alt_names.get(pkg_name)
			if alt and alt != pkg_name:
				try:
					v = _get_dist_version(alt)
					return getattr(v, "version", str(v))
				except Exception:
					pass
			return "installed (version unknown)"

	return "installed (version unknown)"


def pip_freeze() -> str:
	"""Run pip freeze in the same Python interpreter and return output (may be empty on failure)."""
	try:
		result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
		return result.stdout.strip()
	except Exception:
		return ""


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Report versions of common packages.")
	parser.add_argument("--json", "-j", dest="json_out", help="Write JSON output to file")
	parser.add_argument("--freeze", "-f", action="store_true", help="Also include `pip freeze` output")
	parser.add_argument("--packages", "-p", nargs="*", help="Additional package names to check")
	args = parser.parse_args(argv)

	pkgs = list(COMMON_PACKAGES)
	if args.packages:
		pkgs.extend(args.packages)

	report = {}
	for p in pkgs:
		report[p] = get_version_by_import(p)

	if args.freeze:
		report["pip_freeze"] = pip_freeze()

	# Print human readable
	for k, v in report.items():
		if k == "pip_freeze":
			print("\n--- pip freeze ---")
			print(v or "(no output)")
		else:
			print(f"{k}: {v}")

	if args.json_out:
		try:
			with open(args.json_out, "w", encoding="utf-8") as fh:
				json.dump(report, fh, indent=2, ensure_ascii=False)
			print(f"\nWrote JSON output to: {args.json_out}")
		except Exception as e:
			print(f"Failed to write JSON file: {e}", file=sys.stderr)
			return 2

	return 0


if __name__ == "__main__":
	raise SystemExit(main())