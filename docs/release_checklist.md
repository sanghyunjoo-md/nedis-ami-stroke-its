# GitHub and Zenodo v1.1.1 release checklist

Use this checklist for the documentation-alignment release completed after the
Stage 3-D v1.3 cross-audit.

- [ ] Confirm that the final Annals manuscript title and Appendix E1 Figure E6 mapping are aligned.
- [ ] Confirm that `data/raw/` contains only its README.
- [ ] Confirm that `private/` contains only its README.
- [ ] Confirm that no `analysis_outputs/`, `analysis_extension_outputs/`, spreadsheet, CSV, pickle, log, signed document, or local checksum is present.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run the repository exposure scan described in the handoff instructions.
- [ ] Commit the update to `main` without changing or deleting tags `v1.0.0` or `v1.1.0`.
- [ ] Create GitHub tag and release `v1.1.1`.
- [ ] Confirm that Zenodo archived `v1.1.1` as a new version and issued a version DOI.
- [ ] Replace every submission-document reference to v1.1.0 and its version DOI with the v1.1.1 version and DOI.
- [ ] If code changes during review, create another release instead of replacing an existing tag.
