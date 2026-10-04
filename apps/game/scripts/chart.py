"""Beat grid and loudness for each bundled song → public/music/songs.json.

The game generates its notes from this at run time: a note can land on any beat, and the
loudness at that beat decides how dense they are (quiet intros and breakdowns get fewer).

    .venv/bin/python apps/game/scripts/chart.py      # needs librosa
"""

import json
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as sg

ROOT = Path(__file__).resolve().parent.parent
MUSIC = ROOT / "public/music"

# CC0 tracks from Freesound (found through Openverse). Checked for a steady tempo.
SONGS = [
    {"id": "happy-summer-edm", "file": "happy-summer-edm.mp3", "title": "Happy Summer EDM Song", "creator": "Seth_Makes_Sounds",
     "page": "https://freesound.org/people/Seth_Makes_Sounds/sounds/687014", "level": "Normal"},
    {"id": "determined", "file": "determined.mp3", "title": "Determined Video Game Music", "creator": "Seth_Makes_Sounds",
     "page": "https://freesound.org/people/Seth_Makes_Sounds/sounds/685334", "level": "Hard"},
]


def analyze(path: Path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    duration = len(y) / sr
    # Four-on-the-floor tracks: a kick drum on every beat. Find kick onsets in the low band,
    # then the constant tempo and phase that put the most kicks within ±30 ms of a beat.
    b, a = sg.butter(4, 150 / (sr / 2), "low")
    low = sg.filtfilt(b, a, y)
    hop = 128
    env = librosa.onset.onset_strength(y=low, sr=sr, hop_length=hop, n_mels=16, fmax=200)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    peaks = librosa.util.peak_pick(env, pre_max=10, post_max=10, pre_avg=20, post_avg=20,
                                   delta=np.percentile(env, 95) * 0.5, wait=40)
    kicks = t[peaks]
    best = (-1.0, 0.0, 0.0)
    for bpm in np.arange(100, 180, 0.05):
        period = 60 / bpm
        phases = np.linspace(0, period, 160, endpoint=False)
        d = np.abs((kicks[None, :] - phases[:, None] + period / 2) % period - period / 2)
        frac = (d < 0.03).mean(axis=1)
        i = int(frac.argmax())
        if frac[i] > best[0]:
            best = (float(frac[i]), period, float(phases[i]))
    on_grid, period, offset = best
    residual = on_grid
    grid = np.arange(offset, duration - 0.5, period)
    # Loudness per beat, normalised to 0..1 between the quiet and loud ends of the song.
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    per_beat = np.interp(grid + period / 2, times, rms)
    lo, hi = np.percentile(per_beat, 10), np.percentile(per_beat, 95)
    energy = np.clip((per_beat - lo) / (hi - lo + 1e-9), 0, 1)
    return {
        "bpm": round(60 / period, 3),
        "offset": round(offset, 4),
        "duration": round(duration, 2),
        "beats": len(grid),
        "energy": [round(float(e), 2) for e in energy],
        "_kicksOnGrid": round(residual, 2),
    }


def main():
    out = []
    for s in SONGS:
        a = analyze(MUSIC / s["file"])
        print(f'{s["id"]}: {a["bpm"]} BPM, first beat {a["offset"]}s, {a["beats"]} beats, {a.pop("_kicksOnGrid"):.0%} of kicks within ±30 ms of the grid')
        out.append({**s, "license": "CC0 1.0", **a})
    (MUSIC / "songs.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
