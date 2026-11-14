"""Download 16 evenly spaced DeformGen yawonly training videos per task.

For each task this writes the original videos, H.264 transcodes, first-frame
contact sheet, synchronized 4x4 overview video, manifest, and a tar.gz archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download

from .config import ROOT

DATASET_REPO = "Zili2002/DeformGen-Datasets"
DATASET_REVISION = "1190f43fda06af0a04c8f791f7a6488196719401"
EPISODES = (0, 67, 133, 200, 267, 333, 400, 467, 533, 600, 667, 733, 800, 867, 933, 999)
TASK_CASES = {"cloth": "cloth3", "rope": "rope", "sloth": "sloth"}


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _download_source(case: str, episode: int, destination: Path) -> None:
    remote = (
        f"{case}/train1000_yawonly/videos/chunk-000/"
        f"observation.images.front/episode_{episode:06d}.mp4"
    )
    cached = Path(
        hf_hub_download(
            DATASET_REPO,
            remote,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    if not destination.exists() or destination.stat().st_size != cached.stat().st_size:
        shutil.copy2(cached, destination)


def _make_contact_sheet(frames: list[tuple[int, Path]], output: Path) -> None:
    cells = []
    for episode, path in frames:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (150, 26), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"episode {episode:06d}",
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(image)
    rows = [cv2.hconcat(cells[index : index + 4]) for index in range(0, 16, 4)]
    if not cv2.imwrite(str(output), cv2.vconcat(rows)):
        raise RuntimeError(f"Could not write {output}")


def _make_grid(videos: list[tuple[int, Path]], output: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for _, path in videos:
        command += ["-i", str(path)]
    filters = []
    labels = []
    for index, (episode, _) in enumerate(videos):
        filters.append(
            f"[{index}:v]scale=320:180,setsar=1,"
            "drawbox=x=0:y=0:w=150:h=26:color=black@0.75:t=fill,"
            f"drawtext=text='episode {episode:06d}':x=7:y=6:"
            "fontsize=14:fontcolor=white"
            f"[v{index}]"
        )
        labels.append(f"[v{index}]")
    layout = "|".join(f"{(i % 4) * 320}_{(i // 4) * 180}" for i in range(16))
    filters.append("".join(labels) + f"xstack=inputs=16:layout={layout}[grid]")
    command += [
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[grid]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]
    _run(command)


def prepare(task: str, output_root: Path) -> Path:
    case = TASK_CASES[task]
    root = output_root / f"deformgen_{case}_initial_states_yawonly"
    source_dir = root / "videos"
    h264_dir = root / "videos_h264"
    frames_dir = root / "frames"
    for directory in (source_dir, h264_dir, frames_dir):
        directory.mkdir(parents=True, exist_ok=True)

    h264_videos: list[tuple[int, Path]] = []
    frames: list[tuple[int, Path]] = []
    for episode in EPISODES:
        source = source_dir / f"episode_{episode:06d}.mp4"
        h264 = h264_dir / f"episode_{episode:06d}_h264.mp4"
        frame = frames_dir / f"episode_{episode:06d}.jpg"
        _download_source(case, episode, source)
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(h264),
            ]
        )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(h264),
                "-frames:v",
                "1",
                str(frame),
            ]
        )
        h264_videos.append((episode, h264))
        frames.append((episode, frame))

    _make_contact_sheet(frames, root / "contact_sheet_16.jpg")
    _make_grid(h264_videos, root / "episodes_16_grid_h264.mp4")
    archive = root / "episodes_16_h264.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for _, path in h264_videos:
            stream.add(path, arcname=path.name)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "repo_id": DATASET_REPO,
                "revision": DATASET_REVISION,
                "case": case,
                "dataset": f"{case}/train1000_yawonly",
                "camera": "observation.images.front",
                "episodes": list(EPISODES),
                "video_codec": "h264",
            },
            indent=2,
        )
        + "\n"
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_CASES),
        default=list(TASK_CASES),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/reference",
    )
    args = parser.parse_args()
    for task in args.tasks:
        print(prepare(task, args.output_root.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
