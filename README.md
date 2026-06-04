# Predictive Style Matching — Project Page

Project website for *Predictive Style Matching: Natural and Robust Humanoid Locomotion*.

**Live site:** [https://simeon-ned.github.io/predictive-style-matching/](https://simeon-ned.github.io/predictive-style-matching/)

## Local preview

```bash
cd predictive-style-matching
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy (GitHub Pages)

1. Push this repo to `simeon-ned/predictive-style-matching` on GitHub.
2. **Settings → Pages →** source: `master` branch, folder `/ (root)`.
3. Ensure `.nojekyll` stays at the repo root.

## arXiv + RA-L (under review)

IEEE allows [arXiv preprints](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/) for RA-L; they are not treated as prior publication. RA-L uses [double-anonymous review](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/)—reviewers may find your preprint or site, which is common in robotics.

### Use two PDFs

| Version | Authors | Project URL in paper | Where |
|--------|---------|----------------------|--------|
| **Review** | Anonymous | Remove from PDF | IEEE RAS submission |
| **Public** | Full list | OK | arXiv (link on site when posted); PDF via arXiv, not hosted here yet |

The RA-L upload must match the anonymized TeX (no names, no identifiable thanks/URLs). Keep `main.tex` with authors for arXiv; maintain a separate anonymized build for resubmission if needed.

### arXiv upload checklist

- **Category:** `cs.RO` (primary)
- **Title / abstract:** Match the public version exactly
- **PDF:** Deanonymized (full author list; hosted on arXiv only)
- **Comments (metadata):** Neutral only, e.g. `8 pages, 6 figures, 3 tables` — do **not** write “submitted to RA-L” or name the journal
- **Source:** Optional; many authors upload `.tex` without internal-only paths

### Project page (while under review)

**OK**

- Named authors and affiliations
- arXiv link (when ready), videos, code (when ready)
- Link to arXiv after posting

**Avoid**

- “Submitted to / under review at IEEE RA-L”
- `note = {Submitted}` in BibTeX
- Listing the journal as if already accepted

This site uses **“2026 · Preprint”** and arXiv-style BibTeX until acceptance.

### After arXiv posts

1. In `index.html`, replace the `arXiv (soon)` disabled button with a live link to `https://arxiv.org/abs/XXXX.XXXXX`.
2. Uncomment the BibTeX section at the bottom; set `eprint` and `url`, enable the BibTeX header button.
3. Optional: add `citation_arxiv_id` meta tag.

### After RA-L acceptance

- Change venue line to **IEEE Robotics and Automation Letters (RA-L), 2026** (plus vol/issue when known).
- Switch BibTeX to `@article` with journal, DOI, and IEEE copyright notice on the arXiv PDF per IEEE policy.
- Keep arXiv version; add link to the IEEE Xplore version.

## Assets

| Asset | Path | Notes |
|-------|------|--------|
| Figures | `static/images/` | Sync from `../psm_paper/figures/` |
| Hero image | `static/images/snapshots.png` | Background in header |
| Hardware clips | YouTube embeds in `index.html` | Short under Deployment; full at end |
| Favicon | `static/images/favicon.svg` | Humanoid robot icon |

## Paper source

LaTeX: `../psm_paper/`

## Acknowledgments

Built from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
