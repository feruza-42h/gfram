# GFRAM Studio

Desktop demonstration of the dissertation: live recognition, the geometric method step by step,
1:1 / 1:N, protected templates and every experimental result. Interface in Uzbek or Russian.

## Run

The studio is part of the library (`gfram/studio`), version = gfram version.

- Developer machine: `desktop/GFRAM Studio.command` (or `venv/bin/python -m gfram.studio` from the repo root).
- Any computer: `pip install "gfram[studio]"`, then `gfram-studio`.
- One-click installers: `desktop/installers/` — `GFRAM-Studio-macOS.command`, `GFRAM-Studio-Windows.bat`.
  They find or install Python 3.12, create a private environment (`~/.gfram/studio-app`,
  `%LOCALAPPDATA%\GFRAM\StudioApp`), install gfram (a `gfram-*.whl` next to the installer is used if present,
  otherwise PyPI), prepare models, ask for camera access, create a Desktop shortcut and start the app.
  Next double-click just starts it; `--reinstall` reinstalls. Build the kit: wheel + both installers + README.txt.

Updates: «Yangilanishlar» page — installed versions, PyPI check, `pip install --upgrade "gfram[studio]"` with a live
log and restart, geometric model update from gfram.uz. A badge in the sidebar appears when PyPI has a newer version.
From a source checkout the update button is disabled (use git).

Camera capture uses Qt Multimedia (FaceTime HD by default). Press **F** for full screen. Research pages are hidden:
`SHOW_RESEARCH_PAGES = True` in `gfram/studio/app.py` brings them back. Studio data: `~/.gfram/studio/`.

## Seminar plan

1. **Jonli tanish** — camera: 478 points, pose axes, name + probability, per-stage timings; every recognition
   goes to the journal.
2. **Geometrik tahlil** — «LFW dan tasodifiy»: points, mesh, 3D Procrustes shape, 153 classic features, 128 + 512 vectors.
3. **Solishtirish 1:1** — LFW / CPLFW pairs; how each of the 10 fusion features moved the decision.
4. **Ro‘yxatga olish** — guided camera registration (5 poses, auto-capture with quality checks, duplicate check),
   card (name, position, department, ID), people list: edit, rename, add/delete templates, delete person.
5. **Baza fayllari va himoya** — every file on disk with a byte-level image scan (0 photos), what is inside
   persons.pkl, one person's record; proof ①–⑤: enrol from a photo → delete the photo → scan → reload the
   database from the file → another photo of the person is still recognised; key rotation demo; LFW fill.
6. **Identifikatsiya 1:N**, **Tanish jurnali** (CSV export), **Jonli sinov** (LFW 600 pairs ~30 s), **Natijalar**,
   **Ilmiy yangilik**.

Studio files (`~/.gfram/studio`): `database/persons.pkl` (protected templates + AES-encrypted landmarks),
`keys/template.key`, `people.json` (cards), `events.jsonl` (journal). No images anywhere.
