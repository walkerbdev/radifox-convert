import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import List, Optional

from pydicom import dcmread
from pydicom.errors import InvalidDicomError

from radifox.records.hashing import hash_file_dir

from ._version import __version__
from .exec import run_conversion, ExecError
from .lut import LookupTable
from .metadata import Metadata
from .utils import silentremove, mkdir_p, version_check


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_lut_file(
    output_root: Path, metadata: Metadata, lut_file_arg: Optional[Path], no_project_subdir: bool = False
) -> Path:
    """Resolve the LUT file path from an explicit arg or the default location."""
    if lut_file_arg is not None:
        return lut_file_arg
    if no_project_subdir:
        return output_root / (metadata.projectname + "-lut.csv")
    return output_root / metadata.projectname / (metadata.projectname + "-lut.csv")


def _load_manual_names(output_root: Path, metadata: Metadata) -> dict:
    """Load ManualNaming.json if it exists, otherwise return empty dict."""
    manual_json_file = output_root / metadata.dir_to_str() / (metadata.prefix_to_str() + "_ManualNaming.json")
    return json.loads(manual_json_file.read_text()) if manual_json_file.exists() else {}


def _find_first_dicom(directory: Path):
    """Walk a directory to find and read the first valid DICOM file.

    Only one file is needed because patient-level DICOM attributes (PatientID,
    PatientName, etc.) are constant across all files within a subject directory.
    """
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            try:
                ds = dcmread(f, stop_before_pixels=True)
                return ds
            except (InvalidDicomError, Exception):
                continue
    return None


# ---------------------------------------------------------------------------
# radifox-convert  (single mode)
# ---------------------------------------------------------------------------


def _run_single(args: argparse.Namespace) -> None:
    """Run single-subject conversion (original behaviour)."""
    if args.hardlink and args.symlink:
        raise ValueError("Only one of --symlink and --hardlink can be used.")
    linking = "hardlink" if args.hardlink else ("symlink" if args.symlink else None)

    mapping = {"subject_id": "SubjectID", "session_id": "SessionID", "site_id": "SiteID"}
    if args.tms_metafile:
        metadata = Metadata.from_tms_metadata(args.tms_metafile, args.no_project_subdir)
        for argname in ["subject_id", "session_id", "site_id"]:
            if getattr(args, argname) is not None:
                setattr(metadata, mapping[argname], getattr(args, argname))
    else:
        for argname in ["project_id", "subject_id", "session_id"]:
            if getattr(args, argname) is None:
                raise ValueError(
                    "%s is a required argument when no metadata file is provided." % argname
                )
        metadata = Metadata(
            args.project_id,
            args.subject_id,
            args.session_id,
            args.site_id,
            args.no_project_subdir,
        )

    lut_file = _resolve_lut_file(args.output_root, metadata, args.lut_file, args.no_project_subdir)
    manual_names = _load_manual_names(args.output_root, metadata)

    type_dirname = "%s" % "parrec" if args.parrec else "dcm"
    if (args.output_root / metadata.dir_to_str() / type_dirname).exists():
        if args.safe:
            metadata.AttemptNum = 2
            while (args.output_root / metadata.dir_to_str() / type_dirname).exists():
                metadata.AttemptNum += 1
        elif args.force or args.reckless:
            if not args.reckless:
                json_file = (
                    args.output_root
                    / metadata.dir_to_str()
                    / (metadata.prefix_to_str() + "_UnconvertedInfo.json")
                )
                if not json_file.exists():
                    raise ValueError(
                        "Unconverted info file (%s) does not exist for consistency checking. "
                        "Cannot use --force, use --reckless instead." % json_file
                    )
                json_obj = json.loads(json_file.read_text())
                if json_obj["Metadata"]["TMSMetaFileHash"] is not None:
                    if metadata.TMSMetaFileHash is None:
                        raise ValueError(
                            "Previous conversion did not use a TMS metadata file, "
                            "run with --reckless to ignore this error."
                        )
                    if json_obj["Metadata"]["TMSMetaFileHash"] != metadata.TMSMetaFileHash:
                        raise ValueError(
                            "TMS meta data file has changed since last conversion, "
                            "run with --reckless to ignore this error."
                        )
                elif (
                    json_obj["Metadata"]["TMSMetaFileHash"] is None
                    and metadata.TMSMetaFileHash is not None
                ):
                    raise ValueError(
                        "Previous conversion used a TMS metadata file, "
                        "run with --reckless to ignore this error."
                    )
                if hash_file_dir(args.source, False) != json_obj["InputHash"]:
                    raise ValueError(
                        "Source file(s) have changed since last conversion, "
                        "run with --reckless to ignore this error."
                    )
            shutil.rmtree(args.output_root / metadata.dir_to_str() / type_dirname)
            silentremove(args.output_root / metadata.dir_to_str() / "nii")
            for filepath in (args.output_root / metadata.dir_to_str()).glob("*.json"):
                silentremove(filepath)
        else:
            raise RuntimeError(
                "Output directory exists, run with --force to remove outputs and re-run."
            )

    manual_arg = {
        "MagneticFieldStrength": args.field_strength,
        "InstitutionName": args.institution,
    }

    run_conversion(
        args.source,
        args.output_root,
        metadata,
        lut_file,
        args.verbose,
        args.parrec,
        False,
        linking,
        manual_arg,
        args.force_dicom,
        args.anonymize,
        args.date_shift_days,
        manual_names,
        None,
        args.force_derived,
    )


# ---------------------------------------------------------------------------
# radifox-convert  (batch mode)
# ---------------------------------------------------------------------------


def _run_batch(args: argparse.Namespace) -> None:
    """Run batch conversion over subject subdirectories."""
    if not args.source.is_dir():
        raise ValueError("Source must be a directory containing subject subdirectories.")

    subdirs = sorted([d for d in args.source.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError("No subdirectories found in source directory.")

    print("Found %d subdirectories to process." % len(subdirs))

    failed = []
    for i, subdir in enumerate(subdirs, 1):
        print("\n[%d/%d] %s" % (i, len(subdirs), subdir.name))
        success, reason = _process_subject(subdir, args)
        if success:
            print("  OK")
        else:
            print("  FAILED: %s" % reason)
            failed.append((subdir.name, reason))

    print("\n--- Batch conversion complete ---")
    print("Processed: %d, Failed: %d" % (len(subdirs) - len(failed), len(failed)))
    if failed:
        print("\nFailed directories:")
        for name, reason in failed:
            print("  %s: %s" % (name, reason))


def _process_subject(subdir: Path, args: argparse.Namespace):
    """Process a single subject directory for batch conversion.

    Returns (success: bool, error_reason: str or None).
    """
    ds = _find_first_dicom(subdir)
    if ds is None:
        return False, "No valid DICOM files"

    patient_id = getattr(ds, "PatientID", None) or subdir.name
    subject_id = patient_id
    session_id = "1"

    metadata = Metadata(args.project_id, subject_id, session_id, args.site_id, False)
    lut_file = _resolve_lut_file(args.output_root, metadata, args.lut_file)
    manual_names = _load_manual_names(args.output_root, metadata)

    # Check if output already exists
    output_dir = args.output_root / metadata.dir_to_str() / "dcm"
    if output_dir.exists():
        if args.reckless:
            shutil.rmtree(output_dir)
            silentremove(args.output_root / metadata.dir_to_str() / "nii")
            for filepath in (args.output_root / metadata.dir_to_str()).glob("*.json"):
                silentremove(filepath)
        elif args.force:
            return False, "Output exists, use --reckless to overwrite."
        else:
            return False, "Output exists, use --force or --reckless to overwrite."

    try:
        run_conversion(
            subdir,
            args.output_root,
            metadata,
            lut_file,
            args.verbose,
            False,
            False,
            None,
            {"MagneticFieldStrength": None, "InstitutionName": None},
            args.force_dicom,
            args.anonymize,
            args.date_shift_days,
            manual_names,
            None,
        )
        return True, None
    except (ExecError, Exception) as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# CLI entry point: radifox-convert
# ---------------------------------------------------------------------------


def convert(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Source directory/file to convert.")
    parser.add_argument(
        "-o", "--output-root", type=Path, help="Output root directory.", required=True
    )
    parser.add_argument("-l", "--lut-file", type=Path, help="Lookup table file.")
    parser.add_argument("-p", "--project-id", type=str, help="Project ID.")
    parser.add_argument("-s", "--subject-id", type=str, help="Subject ID.")
    parser.add_argument("-e", "--session-id", type=str, help="Session ID.")
    parser.add_argument("--site-id", type=str, help="Site ID.")
    parser.add_argument("--tms-metafile", type=Path, help="TMS metadata file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument(
        "--force", action="store_true", help="Force run even if it would be skipped."
    )
    parser.add_argument(
        "--reckless", action="store_true", help="Force run and overwrite existing data."
    )
    parser.add_argument(
        "--safe", action="store_true", help="Add -N to session ID, if session exists."
    )
    parser.add_argument(
        "--no-project-subdir", action="store_true", help="Do not create project subdirectory."
    )
    parser.add_argument("--parrec", action="store_true", help="Source is PARREC.")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links to source data instead of copying.",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Create hard links to source data instead of copying.",
    )
    parser.add_argument("--institution", type=str, help="Institution name.")
    parser.add_argument("--field-strength", type=int, help="Magnetic field strength.")
    parser.add_argument(
        "--force-dicom", action="store_true", help="Force read DICOM files.", default=False
    )
    parser.add_argument(
        "--force-derived",
        action="store_true",
        help="Convert derived/secondary DICOM series that would normally be skipped.",
        default=False,
    )
    parser.add_argument("--anonymize", action="store_true", help="Anonymize DICOM data.")
    parser.add_argument("--date-shift-days", type=int, help="Number of days to shift dates.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat source as parent directory of subject subdirectories.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    args = parser.parse_args(args)

    for argname in ["source", "output_root", "lut_file", "tms_metafile"]:
        if getattr(args, argname) is not None:
            setattr(args, argname, getattr(args, argname).resolve())

    if args.batch:
        if args.project_id is None:
            raise ValueError("--project-id is required in batch mode.")
        _run_batch(args)
    else:
        _run_single(args)


# ---------------------------------------------------------------------------
# radifox-update
# ---------------------------------------------------------------------------


def update(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Existing RADIFOX Directory to update.")
    parser.add_argument("-l", "--lut-file", type=Path, help="Lookup table file.")
    parser.add_argument(
        "--force", action="store_true", help="Force run even if it would be skipped."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    args = parser.parse_args(args)

    session_id = args.directory.name
    subj_id = args.directory.parent.name

    json_file = args.directory / "_".join([subj_id, session_id, "UnconvertedInfo.json"])
    if not json_file.exists():
        safe_json_file = args.directory / "_".join(
            [subj_id, "-".join(session_id.split("-")[:-1]), "UnconvertedInfo.json"]
        )
        if not safe_json_file.exists():
            raise ValueError("Unconverted info file (%s) does not exist." % json_file)
        json_file = safe_json_file
    json_obj = json.loads(json_file.read_text())

    metadata = Metadata.from_dict(json_obj["Metadata"])
    if session_id != metadata.SessionID:
        metadata.AttemptNum = int(session_id.split("-")[-1])
    # noinspection PyProtectedMember
    output_root = (
        Path(*args.directory.parts[:-2])
        if metadata._NoProjectSubdir
        else Path(*args.directory.parts[:-3])
    )

    lut_file = _resolve_lut_file(output_root, metadata, args.lut_file,
                                 metadata._NoProjectSubdir)  # noinspection PyProtectedMember
    lookup_dict = (
        LookupTable(lut_file, metadata.ProjectID, metadata.SiteID).LookupDict
        if lut_file.exists()
        else {}
    )

    manual_json_file = args.directory / (metadata.prefix_to_str() + "_ManualNaming.json")
    manual_names = json.loads(manual_json_file.read_text()) if manual_json_file.exists() else {}

    if not args.force and (
        version_check(json_obj["__version__"]["radifox"], __version__)
        and json_obj["LookupTable"]["LookupDict"] == lookup_dict
        and json_obj["ManualNames"] == manual_names
    ):
        print(
            "No action required. Software version, LUT dictionary and naming dictionary match for %s."
            % args.directory
        )
        return

    parrec = (args.directory / "parrec").exists()
    type_dir = args.directory / ("%s" % "parrec" if parrec else "dcm")

    mkdir_p(args.directory / "prev")
    for filename in ["nii", "qa", json_file.name]:
        if (args.directory / filename).exists():
            (args.directory / filename).rename(args.directory / "prev" / filename)
    try:
        run_conversion(
            type_dir,
            output_root,
            metadata,
            lut_file,
            args.verbose,
            parrec,
            True,
            None,
            json_obj.get("ManualArgs", {}),
            False,
            False,
            0,
            manual_names,
            json_obj["InputHash"],
        )
    except ExecError:
        logging.info("Exception caught during update. Resetting to previous state.")
        for filename in ["nii", "qa", json_file.name]:
            silentremove(args.directory / filename)
            if (args.directory / "prev" / filename).exists():
                (args.directory / "prev" / filename).rename(args.directory / filename)
    else:
        for dirname in ["stage", "proc"]:
            if (args.directory / dirname).exists():
                (args.directory / dirname / "CHECK").touch()
    silentremove(args.directory / "prev")
