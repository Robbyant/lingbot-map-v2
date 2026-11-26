"""Extract the fixed KITTI 08 example from the official color odometry data."""
import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path


def prepare(source, output):
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    scene = next(scene for scene in manifest["scenes"] if scene["name"] == "kitti08")
    frames = scene["files"]
    names = [f"{frame['frame_id']:06d}.png" for frame in frames]
    if (len(names) != 321 or names != sorted(set(names))
            or any(not re.fullmatch(r"[0-9]{6}\.png", name) for name in names)
            or any(frame["file"] != f"kitti08/{name}" for frame, name in zip(frames, names))):
        raise ValueError("Invalid KITTI 08 frame manifest")
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()} != set(names):
            raise ValueError(f"Choose a new output directory or an exact existing example: {output}")
        for name, frame in zip(names, frames):
            verify((output / name).read_bytes(), frame)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if source.is_dir():
            read = lambda name: (source / name).read_bytes()
        else:
            archive = stack.enter_context(zipfile.ZipFile(source))
            members = set(archive.namelist())
            prefixes = [prefix for prefix in ("dataset/sequences/08/image_2/", "sequences/08/image_2/")
                        if prefix + names[0] in members]
            if len(prefixes) != 1:
                raise ValueError("Expected the official KITTI color odometry archive with sequence 08/image_2")
            read = lambda name: archive.read(prefixes[0] + name)
        temp = stack.enter_context(tempfile.TemporaryDirectory(dir=output.parent))
        staged = Path(temp) / "kitti08"
        staged.mkdir()
        for name, frame in zip(names, frames):
            data = read(name)
            verify(data, frame)
            (staged / name).write_bytes(data)
        staged.rename(output)


def verify(data, frame):
    if len(data) != frame["bytes"] or hashlib.sha256(data).hexdigest() != frame["sha256"]:
        raise ValueError(f"Image differs from the tested frame: {frame['file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="data_odometry_color.zip or the extracted sequences/08/image_2 directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "kitti08")
    args = parser.parse_args()
    prepare(args.source, args.output)
    print(f"Verified 321 ordered KITTI 08 images in {args.output}")
