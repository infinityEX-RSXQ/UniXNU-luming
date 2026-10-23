#!/usr/bin/env python3
"""Static preflight for a UniXNU device repository, runnable on Windows (no macOS tools).

The real ``unixnu`` driver discovers dependencies while it assembles the build tree; that
only happens on the build host. This script performs the same *file-presence and value*
checks a build would hit within the first minutes, before an expensive macOS runner run:

  - board.toml / defconfig parse and required sections
  - every ``[kernel] archives`` builder script resolves (device scripts, then base scripts)
  - ``[kernel.prelink]`` has the keys the prelink step indexes (tool/kernel/slide)
  - ``[userdata]`` partition geometry is actually measured (last_lba > first_lba)
  - ``[payload]`` ADT tool, panic-screen patch, skip-setup tool and kernel args resolve
  - optional package/firmware paths when ``--uefi`` is given
  - defconfig flags that the kernel chain cannot satisfy yet (e.g. BOOT_PROGRESS
    prerequisites, Darwin-22 platform archives whose builders are a TODO(kernel))

Rows are classified as **expected gates** when they are known, documented pre-build
blockers owned by the device/kernel work (currently: iOS 16 prelink values, measured
userdata GPT geometry, kernelcache pin, Darwin-22 kernel archives and defconfig
prerequisites). Everything else is an *unexpected* error. ``--allow-expected-gates``
turns a run whose only errors are gates into exit 0 (useful for reporting); the GitHub
workflow keeps the default strict behaviour so it still fails fast.

It never builds, never downloads an IPSW, never touches a device and never writes into
the inspected repositories.

Usage (Windows):
    python prebuild\\windows\\preflight_inputs.py --device UniXNU-luming
    python prebuild\\windows\\preflight_inputs.py --device UniXNU-luming --base UniXNU-main\\UniXNU-main --steps "kernel userdata payload"
    python prebuild\\windows\\preflight_inputs.py --json prebuild\\windows\\results\\luming-preflight.json
    python prebuild\\windows\\preflight_inputs.py --allow-expected-gates

Exit code: 0 when no errors, 1 when at least one error is found (warnings do not fail);
with ``--allow-expected-gates`` exit 0 when all errors are expected gates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_BASE = WORKSPACE / "UniXNU-main" / "UniXNU-main"
DEFAULT_DEVICE = WORKSPACE / "UniXNU-luming"

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    tomllib = None

# Known, documented pre-build gates. The classifier marks a row as expected when its
# (area, level, message) matches; unknown blockers stay "unexpected" and fail the run.
EXPECTED_GATE_RULES: tuple[tuple[str, str, str, str], ...] = (
    ("kernel", "error", "[kernel.prelink] slide missing", "kernelcache-prelink-analysis"),
    ("kernel", "error", "[kernel.prelink] slide is empty", "kernelcache-prelink-analysis"),
    ("kernel", "warning", "kernelcache_sha256 not pinned", "kernelcache-extraction"),
    ("userdata", "error", "first_lba/last_lba are not a measured partition", "device-gpt-measurement"),
)

# defconfig flag -> builder script that must appear in [kernel] archives when the flag is
# "y"; the base build script links the archive the builder produces and calls test -f on it.
CONFIG_ARCHIVE_BUILDERS: dict[str, str] = {
    "CONFIG_XNU_ARM_DT_PLATFORM": "build-arm-driver.py",
    "CONFIG_XNU_ARMCPU": "build-arm-driver.py",
    "CONFIG_XNU_NATIVE_UFS": "build-arm-driver.py",
    "CONFIG_XNU_NATIVE_RAM_BLOCK": "build-arm-driver.py",
    "CONFIG_XNU_NATIVE_LEGACY_DISPLAY": "build-native-legacy-display.py",
    "CONFIG_XNU_NATIVE_MOBILE_DISPLAY": "build-native-mobile-display.py",
    "CONFIG_XNU_BOOT_PROGRESS": "build-boot-progress.py",
    "CONFIG_XNU_NATIVE_AES": "build-native-aes-bindings.py",
    "CONFIG_XNU_NATIVE_VIRTIO_BLOCK": "build-native-virtio-bindings.py",
}

# Prerequisites enforced by scripts/build-xnu-4k-baseline.sh when the flag is "y".
CONFIG_PREREQUISITES: dict[str, tuple[str, ...]] = {
    # build-xnu-4k-baseline.sh:64-69 ("Boot progress requires the audited ... build")
    "CONFIG_XNU_BOOT_PROGRESS": (
        "CONFIG_XNU_NATIVE_UFS",
        "CONFIG_XNU_OWNER_ROOT_AUDIT",
        "CONFIG_XNU_NATIVE_VIRTIO_BLOCK",
        "CONFIG_XNU_SOURCE_CORECRYPTO14",
    ),
    # build-xnu-4k-baseline.sh:132-136 ("ArmCpu bring-up requires ...")
    "CONFIG_XNU_ARMCPU": (
        "CONFIG_XNU_SOURCE_CORECRYPTO14",
    ),
}


class Result:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, level: str, area: str, message: str, expected: bool = False, gate: str | None = None) -> None:
        self.rows.append({"level": level, "area": area, "message": message,
                          "expected": bool(expected), "gate": gate})

    def error(self, area: str, message: str, expected: bool = False, gate: str | None = None) -> None:
        self.add("error", area, message, expected, gate)

    def warn(self, area: str, message: str, expected: bool = False, gate: str | None = None) -> None:
        self.add("warning", area, message, expected, gate)

    def ok(self, area: str, message: str) -> None:
        self.add("ok", area, message)

    def classify_expected(self) -> None:
        """Mark documented pre-build gates after the fact; anything else stays unexpected."""
        for row in self.rows:
            if row["expected"] or row["level"] == "ok":
                continue
            for area, level, needle, gate in EXPECTED_GATE_RULES:
                if row["area"] == area and row["level"] == level and needle in str(row["message"]):
                    row["expected"] = True
                    row["gate"] = gate
                    break

    @property
    def errors(self) -> list[dict[str, object]]:
        return [row for row in self.rows if row["level"] == "error"]

    @property
    def unexpected_errors(self) -> list[dict[str, object]]:
        return [row for row in self.rows if row["level"] == "error" and not row["expected"]]

    @property
    def warnings(self) -> list[dict[str, object]]:
        return [row for row in self.rows if row["level"] == "warning"]

    @property
    def gates(self) -> list[dict[str, object]]:
        return [row for row in self.rows if row["expected"]]


def load_toml(path: Path):
    if tomllib is None:
        raise RuntimeError("Python 3.11+ is required (tomllib)")
    with path.open("rb") as stream:
        return tomllib.load(stream)


def script_candidates(base: Path, device: Path, name: str):
    """(relative path, candidates) the way unixnu's tree assembles scripts: base first, then device (device wins)."""
    relative = Path(name)
    # board.toml writes both "build-x.py" and "scripts/build-x.py"; the tree stores them under scripts/.
    if relative.parts and relative.parts[0] == "scripts":
        relative = Path(*relative.parts[1:])
    return [device / "scripts" / relative, base / "scripts" / relative]


def resolve_script(base: Path, device: Path, name: str):
    found = []
    for candidate in script_candidates(base, device, name):
        if candidate.is_file():
            found.append(candidate)
    return found


def parse_defconfig(path: Path) -> dict[str, object]:
    """Minimal mirror of unixnu's CONFIG_XNU_* parser (y / n / number / "string")."""
    flags: dict[str, object] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return flags
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        if not separator or not name.startswith("CONFIG_"):
            continue
        value = value.strip()
        if value == "y":
            flags[name] = True
        elif value == "n":
            flags[name] = False
        elif len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            flags[name] = value[1:-1]
        else:
            try:
                flags[name] = int(value, 0)
            except ValueError:
                flags[name] = value
    return flags


def defconfig_gate_checks(result: Result, kernel_spec: dict, config_flags: dict[str, object]) -> None:
    """Predict the kernel-chain failures that the defconfig itself causes (TODO(kernel) gates)."""
    if not config_flags:
        return
    enabled = sorted(name for name, value in config_flags.items() if value is True)
    for flag in enabled:
        prerequisites = CONFIG_PREREQUISITES.get(flag)
        if not prerequisites:
            continue
        missing = [name for name in prerequisites if config_flags.get(name) is not True]
        if missing:
            result.error(
                "kernel",
                f"defconfig enables {flag}=y but {', '.join(missing)} are not y: "
                "scripts/build-xnu-4k-baseline.sh stops before compiling "
                f"({flag} prerequisites); disable {flag} or enable the audited prerequisites (TODO(kernel))",
                expected=True, gate="kernel-config-prerequisites")
    builders = {entry.split()[0] for entry in (kernel_spec.get("archives") or []) if entry.split()}
    missing_builders = sorted({(flag, script) for flag, script in CONFIG_ARCHIVE_BUILDERS.items()
                               if config_flags.get(flag) is True and script not in builders})
    if missing_builders:
        result.error(
            "kernel",
            "defconfig enables flags whose Darwin-22 archive builders are not in [kernel] archives: "
            + ", ".join(f"{flag}->{script}" for flag, script in missing_builders)
            + " -- the XNU link fails on the missing .a after the compile (TODO(kernel); "
            "documented in board.toml/config comments)",
            expected=True, gate="darwin22-kernel-archives")


def check_rows(result: Result, area: str, count: int, description: str) -> None:
    if count == 0:
        result.ok(area, description)


def run(args: argparse.Namespace) -> int:
    result = Result()
    base = args.base.resolve()
    device = args.device.resolve()
    steps = args.steps.split()

    # ---- repositories and board file ------------------------------------
    if not (base / "unixnu").is_file() or not (base / "scripts" / "unixnu" / "cli.py").is_file():
        result.error("base", f"{base} is not a UniXNU base repository (no unixnu driver)")
        return finish(result, args)
    result.ok("base", f"UniXNU base repository: {base}")
    board_path = device / "board.toml"
    if not board_path.is_file():
        result.error("device", f"{board_path} missing")
        return finish(result, args)
    try:
        board = load_toml(board_path)
    except Exception as error:  # noqa: BLE001 - report any parse error
        result.error("device", f"board.toml does not parse: {error}")
        return finish(result, args)
    result.ok("device", f"device repository: {device}")

    board_section = board.get("board", {})
    name = board_section.get("name")
    if not name:
        result.error("board", "[board] name is missing")
    else:
        result.ok("board", f"[board] name = {name!r} -> out/{name}/")
    if board_section.get("example"):
        result.error("board", "[board] example = true: build and bootimg refuse an example repository")
    if not (device / "package").is_dir():
        result.error("board", "package/ missing: the driver refuses a tree without a package module")

    # ---- defconfig -------------------------------------------------------
    kernel_spec = board.get("kernel", {})
    defconfig = kernel_spec.get("defconfig")
    config_flags: dict[str, object] = {}
    if not defconfig:
        result.error("defconfig", "[kernel] defconfig is missing")
    elif not (device / "configs" / defconfig).is_file():
        result.error("defconfig", f"configs/{defconfig} not found (unixnu defconfig would fail)")
    else:
        config_flags = parse_defconfig(device / "configs" / defconfig)
        enabled_count = sum(1 for value in config_flags.values() if value is True)
        result.ok("defconfig", f"configs/{defconfig} present, {enabled_count} enabled CONFIG_XNU_* flags")
    if args.defconfig and defconfig and args.defconfig != defconfig:
        result.error("defconfig", f"workflow input defconfig={args.defconfig} differs from board.toml {defconfig}")

    # ---- kernel ----------------------------------------------------------
    if "kernel" in steps:
        archives = kernel_spec.get("archives", [])
        missing = []
        for entry in archives:
            script = entry.split()[0]
            if not resolve_script(base, device, script):
                missing.append(script)
        if missing:
            result.error("kernel", "[kernel] archives scripts missing from device/scripts and base/scripts: "
                         + ", ".join(sorted(set(missing)))
                         + " -- the kernel step fails in build_archives() before the link")
        else:
            result.ok("kernel", f"all {len(archives)} [kernel] archives scripts resolve")
        defconfig_gate_checks(result, kernel_spec, config_flags)
        prelink = kernel_spec.get("prelink")
        if not prelink:
            result.error("kernel", "[kernel.prelink] section missing: the prelink step refuses to run")
        else:
            for key in ("tool", "kernel", "slide"):
                if key not in prelink:
                    result.error("kernel", f"[kernel.prelink] {key} missing: kernel.prelink() would fail")
            if "slide" in prelink and not str(prelink["slide"]).strip():
                result.error("kernel", "[kernel.prelink] slide is empty")
            if "kernelcache_sha256" not in prelink:
                result.warn("kernel", "[kernel.prelink] kernelcache_sha256 not pinned: any extracted kernelcache is accepted")
            if "roots" in prelink and not isinstance(prelink["roots"], list):
                result.error("kernel", "[kernel.prelink] roots must be an array")
            if "early_security_init" in prelink and not str(prelink["early_security_init"]).strip():
                result.error("kernel", "[kernel.prelink] early_security_init is empty")
            for key in ("slide", "tool", "kernel"):
                if key in prelink:
                    result.ok("kernel", f"[kernel.prelink] {key} = {prelink[key]!r}")

    # ---- userdata --------------------------------------------------------
    if "userdata" in steps:
        userdata = board.get("userdata")
        if not userdata:
            result.error("userdata", "[userdata] section missing")
        else:
            first, last = userdata.get("first_lba"), userdata.get("last_lba")
            if not isinstance(first, int) or not isinstance(last, int) or last <= first:
                result.error("userdata", f"[userdata] first_lba/last_lba are not a measured partition ({first!r}/{last!r}): "
                             "the image is larger than the partition and the step fails after extracting ~7.5 GiB; "
                             "measure the GPT entry named 'userdata' and fill board.toml")
            else:
                lba = userdata.get("lba_bytes", 4096)
                capacity = (last - first + 1) * lba
                result.ok("userdata", f"[userdata] partition = {capacity / 2**30:.2f} GiB ({first:#x}..{last:#x}, {lba} B/LBA)")
            windows = userdata.get("windows") or []
            if not windows:
                result.error("userdata", "[userdata] windows is empty")
            for window in windows:
                if not window.get("name") or not window.get("component"):
                    result.error("userdata", f"[userdata] window needs name and component: {window!r}")
            if windows:
                result.ok("userdata", f"{len(windows)} userdata windows: " + ", ".join(str(window.get("name")) for window in windows))

    # ---- payload ---------------------------------------------------------
    if "payload" in steps:
        payload = board.get("payload")
        if not payload:
            result.error("payload", "[payload] section missing")
        else:
            adt = payload.get("adt") or {}
            tool = adt.get("tool")
            if not tool:
                result.error("payload", "[payload.adt] tool missing")
            elif not resolve_script(base, device, tool):
                result.error("payload", f"[payload.adt] tool {tool} not found in device/scripts or base/scripts")
            else:
                result.ok("payload", f"[payload.adt] tool {tool} resolves")
            if adt.get("unique_chip_id") == "required":
                result.ok("payload", "[payload.adt] unique_chip_id = required -> the workflow needs the UNIXNU_UNIQUE_CHIP_ID secret")
            if payload.get("panic_screen"):
                patcher = resolve_script(base, device, "patch-xnu-entry-probe.py")
                if not patcher:
                    result.error("payload", "[payload] panic_screen = true but scripts/patch-xnu-entry-probe.py exists in neither "
                                 "device/scripts nor base/scripts: the payload step fails after copying the kernel")
                else:
                    result.ok("payload", "panic_screen patch script resolves: " + ", ".join(str(p) for p in patcher))
            skip_tool = payload.get("skip_setup_tool")
            if skip_tool and not resolve_script(base, device, skip_tool):
                result.error("payload", f"[payload] skip_setup_tool {skip_tool} not found (only used with --skip-setup)")
            volume = str(payload.get("volume", "UNIXNU"))
            if len(volume) > 11:
                result.error("payload", f"[payload] volume {volume!r} is longer than 11 characters")
            size = payload.get("size_mib", 192)
            if not isinstance(size, int) or size <= 0:
                result.error("payload", f"[payload] size_mib invalid: {size!r}")
            arguments = payload.get("kernel_args") or []
            line = " ".join(str(item) for item in arguments)
            if len(line) > 1000:
                result.error("payload", f"[payload] kernel_args are {len(line)} bytes; the limit is 1000")
            for value in arguments:
                import re
                if not re.fullmatch(r"[A-Za-z0-9_.\-]+(=[A-Za-z0-9_.,\-]+)?", str(value)):
                    result.error("payload", f"[payload] kernel_args: unsupported argument {value!r}")
            if not isinstance(payload.get("dram_size"), (str, int)):
                result.error("payload", "[payload] dram_size missing")
            for fixup in payload.get("text_fixups") or []:
                if not isinstance(fixup, str) or len(fixup) >= 160:
                    result.error("payload", f"[payload] text_fixups entry malformed: {fixup!r}")
            result.ok("payload", f"[payload] volume={volume!r} size_mib={size} kernel_args={len(arguments)}")

    # ---- firmware / package (only when the uefi checkout is given) -------
    firmware = board.get("firmware")
    if firmware:
        for script in firmware.get("preflight") or []:
            if not resolve_script(base, device, script):
                result.error("firmware", f"[firmware] preflight script {script} not found (firmware step fails)")
            else:
                result.ok("firmware", f"[firmware] preflight {script} resolves")
        if args.uefi:
            uefi = args.uefi.resolve()
            config = firmware.get("config")
            config_path = uefi / "Resources" / "Configs" / f"{config}.toml" if config else None
            if not uefi.is_dir():
                result.error("firmware", f"--uefi {uefi} does not exist")
            elif config_path and not config_path.is_file():
                result.error("firmware", f"{config_path} missing: is the checkout a Mu-Silicium tree for luming?")
            elif config_path:
                result.ok("firmware", f"Mu-Silicium config: {config_path}")
            dsc = firmware.get("dsc")
            if dsc and not (uefi / dsc).is_file():
                result.error("firmware", f"[firmware] dsc {dsc} not found under {uefi}")
            mkbootimg = uefi / "Resources" / "Scripts" / "mkbootimg.py"
            if not mkbootimg.is_file():
                result.error("firmware", f"{mkbootimg} missing: bootimg needs a Mu-Silicium checkout")
            else:
                result.ok("firmware", "Resources/Scripts/mkbootimg.py present (bootimg)")
        package = firmware.get("packages") or []
        if not package:
            result.warn("firmware", "[firmware] packages is empty")
    if "package" in steps and not (base / "package" / "build.sh").is_file():
        result.error("package", "base package/build.sh missing: the package step cannot build")

    return finish(result, args)


def finish(result: Result, args: argparse.Namespace) -> int:
    result.classify_expected()
    errors = len(result.errors)
    unexpected_errors = len(result.unexpected_errors)
    gates = result.gates
    warnings = len(result.warnings)
    gate_ids = sorted({str(row["gate"]) for row in gates})
    build_ready = errors == 0
    exit_code = 0 if build_ready or (args.allow_expected_gates and unexpected_errors == 0) else 1
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    document = {
        "schema": 2,
        "generated_utc": stamp,
        "device": str(args.device.resolve()),
        "base": str(args.base.resolve()),
        "steps": args.steps.split(),
        "errors": errors,
        "unexpected_errors": unexpected_errors,
        "expected_gates": len(gates),
        "gates_by_id": gate_ids,
        "warnings": warnings,
        "ok": build_ready,
        "build_ready": build_ready,
        "allow_expected_gates": bool(args.allow_expected_gates),
        "exit_code": exit_code,
        "rows": result.rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in result.rows:
        if row["expected"]:
            mark = "GATE "
        else:
            mark = {"error": "ERROR", "warning": "WARN ", "ok": "ok   "}[row["level"]]
        suffix = f"  [{row['gate']}]" if row["expected"] else ""
        print(f"[{mark}] {row['area']:9} {row['message']}{suffix}")
    if build_ready:
        verdict = "ready for the macOS runner steps that only need these inputs"
    elif unexpected_errors == 0:
        verdict = (f"not ready by design: {len(gates)} expected pre-build gate(s) remain "
                   f"({', '.join(gate_ids)}); the runner would fail fast at these gates")
    else:
        verdict = f"{unexpected_errors} unexpected error(s) must be fixed before triggering the workflow"
    print(f"\npreflight_inputs: {unexpected_errors} unexpected error(s), {len(gates)} expected gate(s), "
          f"{warnings} warning(s) -> {verdict}")
    if args.json:
        print(f"result JSON: {args.json}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=Path, default=DEFAULT_DEVICE, help=f"device repository (default: {DEFAULT_DEVICE})")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help=f"UniXNU base repository (default: {DEFAULT_BASE})")
    parser.add_argument("--steps", default="kernel userdata payload", help='steps to check, default "kernel userdata payload"')
    parser.add_argument("--defconfig", default="", help="expected defconfig name (workflow input)")
    parser.add_argument("--uefi", type=Path, help="Mu-Silicium checkout (enables the firmware/bootimg checks)")
    parser.add_argument("--json", type=Path, help="write the result as JSON")
    parser.add_argument("--allow-expected-gates", action="store_true",
                        help="exit 0 when the only errors are documented pre-build gates (reporting mode)")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
