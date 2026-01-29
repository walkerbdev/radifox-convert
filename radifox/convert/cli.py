import argparse
import json
import logging
import shutil
from pathlib import Path

from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset
from pydicom.errors import InvalidDicomError

from radifox.records.hashing import hash_file_dir

from ._version import __version__
from .anondb import AnonDB
from .deanon import deanonymize_subject
from .exec import ExecError, run_conversion
from .lut import LookupTable
from .metadata import Metadata
from .utils import mkdir_p, silentremove, version_check

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_lut_file(
    output_root: Path, metadata: Metadata, lut_file_arg: Path | None, no_project_subdir: bool = False
) -> Path:
    """Resolve the LUT file path from an explicit arg or the default location."""
    if lut_file_arg is not None:
        return lut_file_arg
    if no_project_subdir:
        return output_root / (metadata.projectname + "-lut.csv")
    return output_root / metadata.projectname / (metadata.projectname + "-lut.csv")


def _load_manual_names(output_root: Path, metadata: Metadata) -> dict[str, str]:
    """Load ManualNaming.json if it exists, otherwise return empty dict."""
    manual_json_file = output_root / metadata.dir_to_str() / (metadata.prefix_to_str() + "_ManualNaming.json")
    return json.loads(manual_json_file.read_text()) if manual_json_file.exists() else {}


def _extract_patient_info(ds: Dataset | FileDataset) -> dict[str, str | None]:
    """Extract patient-level DICOM attributes from a pydicom Dataset."""
    return {
        "patient_id": getattr(ds, "PatientID", None),
        "patient_name": str(getattr(ds, "PatientName", "")) or None,
        "patient_birth_date": getattr(ds, "PatientBirthDate", None),
        "patient_sex": getattr(ds, "PatientSex", None),
        "study_uid": getattr(ds, "StudyInstanceUID", None),
        "institution_name": getattr(ds, "InstitutionName", None),
    }


def _find_first_dicom(directory: Path) -> Dataset | FileDataset | None:
    """Walk a directory to find and read the first valid DICOM file.

    Only one file is needed because patient-level DICOM attributes (PatientID,
    PatientName, PatientBirthDate, etc.) are constant across all series and
    files within a subject's directory.
    """
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            try:
                ds = dcmread(f, stop_before_pixels=True)
                return ds
            except (InvalidDicomError, Exception):
                continue
    return None


def _check_output_exists(
    output_root: Path, metadata: Metadata, type_dirname: str, force: bool, reckless: bool
) -> str | None:
    """Check if output already exists and handle force/reckless.

    Returns None if OK to proceed, or a skip-reason string.
    """
    output_dir = output_root / metadata.dir_to_str() / type_dirname
    if not output_dir.exists():
        return None

    if reckless:
        shutil.rmtree(output_dir)
        silentremove(output_root / metadata.dir_to_str() / "nii")
        for filepath in (output_root / metadata.dir_to_str()).glob("*.json"):
            silentremove(filepath)
        return None

    if force:
        return "Output exists, use --reckless to overwrite."

    return "Output exists, use --force or --reckless to overwrite."


# ---------------------------------------------------------------------------
# radifox-convert  (single mode)
# ---------------------------------------------------------------------------


def _run_single(args: argparse.Namespace) -> None:
    """Run single-subject conversion."""
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
                raise ValueError(f"{argname} is a required argument when no metadata file is provided.")
        metadata = Metadata(
            args.project_id,
            args.subject_id,
            args.session_id,
            args.site_id,
            args.no_project_subdir,
        )

    lut_file = _resolve_lut_file(args.output_root, metadata, args.lut_file, args.no_project_subdir)
    manual_names = _load_manual_names(args.output_root, metadata)

    type_dirname = "{}".format("parrec") if args.parrec else "dcm"
    if (args.output_root / metadata.dir_to_str() / type_dirname).exists():
        if args.safe:
            # Metadata.AttemptNum is typed as None but accepts int at runtime
            metadata.AttemptNum = 2  # type: ignore[assignment]
            while (args.output_root / metadata.dir_to_str() / type_dirname).exists():
                metadata.AttemptNum += 1  # type: ignore[assignment,operator]
        elif args.force or args.reckless:
            if not args.reckless:
                json_file = (
                    args.output_root / metadata.dir_to_str() / (metadata.prefix_to_str() + "_UnconvertedInfo.json")
                )
                if not json_file.exists():
                    raise ValueError(
                        f"Unconverted info file ({json_file}) does not exist for consistency checking. "
                        "Cannot use --force, use --reckless instead."
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
                elif json_obj["Metadata"]["TMSMetaFileHash"] is None and metadata.TMSMetaFileHash is not None:
                    raise ValueError(
                        "Previous conversion used a TMS metadata file, run with --reckless to ignore this error."
                    )
                if hash_file_dir(args.source, False) != json_obj["InputHash"]:
                    raise ValueError(
                        "Source file(s) have changed since last conversion, run with --reckless to ignore this error."
                    )
            shutil.rmtree(args.output_root / metadata.dir_to_str() / type_dirname)
            silentremove(args.output_root / metadata.dir_to_str() / "nii")
            for filepath in (args.output_root / metadata.dir_to_str()).glob("*.json"):
                silentremove(filepath)
        else:
            raise RuntimeError("Output directory exists, run with --force to remove outputs and re-run.")

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
    anonymize = args.anon_db is not None

    if not args.source.is_dir():
        raise ValueError("Source must be a directory containing subject subdirectories.")

    subdirs = sorted([d for d in args.source.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError("No subdirectories found in source directory.")

    print(f"Found {len(subdirs)} subdirectories to process.")
    if anonymize:
        print(f"Anonymization enabled (mapping database: {args.anon_db})")

    db = None
    if anonymize:
        db = AnonDB(args.anon_db)

    failed = []
    try:
        for i, subdir in enumerate(subdirs, 1):
            print(f"\n[{i}/{len(subdirs)}] {subdir.name}")
            success, reason = _process_subject(subdir, args, db, anonymize)
            if success:
                print("  OK")
            else:
                print(f"  FAILED: {reason}")
                failed.append((subdir.name, reason))
    finally:
        if db is not None:
            db.close()

    print("\n--- Batch conversion complete ---")
    print(f"Processed: {len(subdirs) - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("\nFailed directories:")
        for name, reason in failed:
            print(f"  {name}: {reason}")


def _process_subject(
    subdir: Path, args: argparse.Namespace, db: AnonDB | None, anonymize: bool
) -> tuple[bool, str | None]:
    """Process a single subject directory for batch conversion.

    Returns (success: bool, error_reason: str or None).
    """
    ds = _find_first_dicom(subdir)
    if ds is None:
        return False, "No valid DICOM files"

    info = _extract_patient_info(ds)
    patient_id = info["patient_id"] or subdir.name

    if anonymize:
        # Assert needed because mypy can't infer db is not None when anonymize=True
        # Without this: "Item 'None' of 'AnonDB | None' has no attribute 'get_or_create_subject'"
        assert db is not None
        anon_id = db.get_or_create_subject(
            patient_id=patient_id,
            patient_name=info["patient_name"],
            patient_birth_date=info["patient_birth_date"],
            patient_sex=info["patient_sex"],
            date_shift_days=args.date_shift_days,
        )
        session_id = db.add_session(
            anon_id=anon_id,
            source_path=str(subdir),
            original_study_uid=info["study_uid"],
            institution_name=info["institution_name"],
        )
        subject_id = anon_id
    else:
        subject_id = patient_id
        session_id = "1"

    metadata = Metadata(args.project_id, subject_id, session_id, args.site_id, False)
    lut_file = _resolve_lut_file(args.output_root, metadata, args.lut_file)
    manual_names = _load_manual_names(args.output_root, metadata)

    skip_reason = _check_output_exists(args.output_root, metadata, "dcm", args.force, args.reckless)
    if skip_reason is not None:
        return False, f"Output exists: {skip_reason}"

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
            anonymize,
            args.date_shift_days,
            manual_names,
            None,
        )
        if anonymize:
            assert db is not None
            db.commit()
        return True, None
    except (ExecError, Exception) as e:
        if anonymize:
            assert db is not None
            db.rollback()
        return False, str(e)


# ---------------------------------------------------------------------------
# radifox-convert  (deanonymize mode)
# ---------------------------------------------------------------------------


def _run_deanonymize(args: argparse.Namespace) -> None:
    """Reverse anonymization using the mapping database."""
    project_id = args.project_id.upper()
    project_dir = args.output_root / project_id.lower()

    if not project_dir.exists():
        raise ValueError(f"Project directory does not exist: {project_dir}")

    with AnonDB(args.anon_db) as db:
        subjects = db.get_all_subjects()
        if args.subject:
            subjects = [s for s in subjects if s.patient_id == args.subject]
            if not subjects:
                raise ValueError(f"Patient ID '{args.subject}' not found in database.")

        for subject in subjects:
            sessions = db.get_sessions_for_subject(subject.anon_id)
            deanonymize_subject(project_dir, project_id, subject, sessions)

    print(f"\n--- De-anonymized {len(subjects)} subject(s) ---")


# ---------------------------------------------------------------------------
# CLI entry point: radifox-convert
# ---------------------------------------------------------------------------


def convert(argv: list[str] | None = None) -> None:
    """CLI entry point for radifox-convert. Converts DICOM or PARREC sources
    into the RADIFOX project structure. Supports single, batch, and
    deanonymize modes."""
    parser = argparse.ArgumentParser(
        description="Convert DICOM/PARREC sources. Use --batch for multiple subjects, "
        "--deanonymize to reverse anonymization."
    )
    parser.add_argument("source", type=Path, nargs="?", help="Source directory/file to convert.")
    parser.add_argument("-o", "--output-root", type=Path, help="Output root directory.", required=True)
    parser.add_argument("-l", "--lut-file", type=Path, help="Lookup table file.")
    parser.add_argument("-p", "--project-id", type=str, help="Project ID.")
    parser.add_argument("-s", "--subject-id", type=str, help="Subject ID.")
    parser.add_argument("-e", "--session-id", type=str, help="Session ID.")
    parser.add_argument("--site-id", type=str, help="Site ID.")
    parser.add_argument("--tms-metafile", type=Path, help="TMS metadata file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--force", action="store_true", help="Force run even if it would be skipped.")
    parser.add_argument("--reckless", action="store_true", help="Force run and overwrite existing data.")
    parser.add_argument("--safe", action="store_true", help="Add -N to session ID, if session exists.")
    parser.add_argument("--no-project-subdir", action="store_true", help="Do not create project subdirectory.")
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
    parser.add_argument("--force-dicom", action="store_true", help="Force read DICOM files.", default=False)
    parser.add_argument(
        "--force-derived",
        action="store_true",
        help="Convert derived/secondary DICOM series that would normally be skipped.",
        default=False,
    )
    parser.add_argument("--anonymize", action="store_true", help="Anonymize DICOM data.")
    parser.add_argument("--date-shift-days", type=int, help="Number of days to shift dates.")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    # Batch mode arguments
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat source as parent directory of subject subdirectories.",
    )
    parser.add_argument(
        "--anon-db",
        type=Path,
        help="Path to SQLite anonymization mapping database.",
    )
    # Deanonymize mode arguments
    parser.add_argument(
        "--deanonymize",
        action="store_true",
        help="Reverse anonymization using the mapping database (requires --anon-db, --project-id).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="De-anonymize only this patient ID (only with --deanonymize).",
    )

    args = parser.parse_args(argv)

    # Resolve paths
    path_args = ["source", "output_root", "lut_file", "tms_metafile", "anon_db"]
    for argname in path_args:
        if getattr(args, argname, None) is not None:
            setattr(args, argname, getattr(args, argname).resolve())

    # -----------------------------------------------------------------------
    # Mode dispatch and validation
    # -----------------------------------------------------------------------

    if args.deanonymize:
        # Deanonymize mode: requires --anon-db and --project-id, rejects everything else
        if args.anon_db is None:
            raise ValueError("--deanonymize requires --anon-db.")
        if args.project_id is None:
            raise ValueError("--deanonymize requires --project-id.")
        _run_deanonymize(args)
        return

    # --subject is only valid with --deanonymize
    if args.subject:
        raise ValueError("--subject is only valid with --deanonymize.")

    # source is required for convert/batch modes
    if args.source is None:
        raise ValueError("source is required (omit only with --deanonymize).")

    # --anon-db implies --batch
    batch_mode = args.batch or args.anon_db is not None

    if args.date_shift_days is not None and args.anon_db is None:
        raise ValueError("--date-shift-days requires --anon-db.")

    if batch_mode:
        if args.project_id is None:
            raise ValueError("--project-id is required in batch mode.")
        _run_batch(args)
    else:
        _run_single(args)


# ---------------------------------------------------------------------------
# radifox-update
# ---------------------------------------------------------------------------


def update(argv: list[str] | None = None) -> None:
    """CLI entry point for radifox-update. Re-runs naming and conversion on an
    existing RADIFOX session directory using updated LUT or software version."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Existing RADIFOX Directory to update.")
    parser.add_argument("-l", "--lut-file", type=Path, help="Lookup table file.")
    parser.add_argument("--force", action="store_true", help="Force run even if it would be skipped.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    args = parser.parse_args(argv)

    session_id = args.directory.name
    subj_id = args.directory.parent.name

    json_file = args.directory / "_".join([subj_id, session_id, "UnconvertedInfo.json"])
    if not json_file.exists():
        safe_json_file = args.directory / "_".join(
            [subj_id, "-".join(session_id.split("-")[:-1]), "UnconvertedInfo.json"]
        )
        if not safe_json_file.exists():
            raise ValueError(f"Unconverted info file ({json_file}) does not exist.")
        json_file = safe_json_file
    json_obj = json.loads(json_file.read_text())

    metadata = Metadata.from_dict(json_obj["Metadata"])
    if session_id != metadata.SessionID:
        # Metadata.AttemptNum is typed as None but accepts int at runtime
        metadata.AttemptNum = int(session_id.split("-")[-1])  # type: ignore[assignment]
    # noinspection PyProtectedMember
    output_root = Path(*args.directory.parts[:-2]) if metadata._NoProjectSubdir else Path(*args.directory.parts[:-3])

    # noinspection PyProtectedMember
    lut_file = _resolve_lut_file(output_root, metadata, args.lut_file, metadata._NoProjectSubdir)
    # SiteID can be None but LookupTable handles it at runtime
    lookup_dict = LookupTable(lut_file, metadata.ProjectID, metadata.SiteID).LookupDict if lut_file.exists() else {}  # type: ignore[arg-type]

    manual_json_file = args.directory / (metadata.prefix_to_str() + "_ManualNaming.json")
    manual_names = json.loads(manual_json_file.read_text()) if manual_json_file.exists() else {}

    if not args.force and (
        version_check(json_obj["__version__"]["radifox"], __version__)
        and json_obj["LookupTable"]["LookupDict"] == lookup_dict
        and json_obj["ManualNames"] == manual_names
    ):
        print(f"No action required. Software version, LUT dictionary and naming dictionary match for {args.directory}.")
        return

    parrec = (args.directory / "parrec").exists()
    type_dir = args.directory / ("{}".format("parrec") if parrec else "dcm")

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
