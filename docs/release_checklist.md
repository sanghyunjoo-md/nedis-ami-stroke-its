# GitHub and Zenodo v1.1.0 release checklist

Use this checklist for the follow-up release that adds the analysis extension.

- [ ] Upload only the changed or added files from this package to the existing repository.
- [ ] Confirm that `data/raw/` contains only its README.
- [ ] Confirm that `private/` contains only its README.
- [ ] Confirm that no `analysis_outputs/`, `analysis_extension_outputs/`, spreadsheet, CSV, pickle, log, signed document, or local checksum is present.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run the repository exposure scan described in the handoff instructions.
- [ ] Commit the update to `main` without changing or deleting tag `v1.0.0`.
- [ ] Create GitHub tag and release `v1.1.0`.
- [ ] Confirm that Zenodo archived `v1.1.0` as a new version and issued a version DOI.
- [ ] Add the GitHub URL and Zenodo DOI to the manuscript and submission system.
- [ ] If code changes during review, create another release instead of replacing an existing tag.
