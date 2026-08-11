# GitHub and Zenodo release checklist

Complete this checklist while the GitHub repository is still private.

- [ ] Upload only the contents of this package.
- [ ] Confirm that `data/raw/` contains only its README.
- [ ] Confirm that `private/` contains only its README.
- [ ] Confirm that no `analysis_outputs/`, spreadsheet, CSV, pickle, log, or signed document is present.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run the repository exposure scan described in the handoff instructions.
- [ ] Obtain coauthor approval for the public code release and MIT license.
- [ ] Change the GitHub repository visibility from Private to Public.
- [ ] Connect the public repository to Zenodo before creating the release.
- [ ] Create GitHub tag and release `v1.0.0`.
- [ ] Confirm that Zenodo archived `v1.0.0` and issued a DOI.
- [ ] Add the GitHub URL and Zenodo DOI to the manuscript and submission system.
- [ ] If code changes during review, create a new release instead of replacing `v1.0.0`.

