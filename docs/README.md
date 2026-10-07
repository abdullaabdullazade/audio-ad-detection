# Results dashboard

[`index.html`](index.html) is the GitHub Pages site at
https://abdullaabdullazade.github.io/audio-ad-detection/.

It shows the official detection results for 2026-06-11, 2026-06-12, and
2026-06-13: run statistics, charts, the synthetic benchmark, and all 684 detected
airings with an audio player for each.

- `data/detections.json` — every detection plus the batch run stats and synthetic report the page reads.
- `data/<day>.csv` — per-day detection list: clip, recording, advert, offset, clock time, duration, confidence, score, scale.
- `data/<day>/*.mp3` — the detected airing with 2 s of context on each side. These are short excerpts, not the full recordings.
