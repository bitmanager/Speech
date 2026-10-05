# SPDX-License-Identifier: Apache-2.0
"""Convert local Balalaika tar/parquet pairs to ordinary Lhotse ASR cuts.

No word timestamps are fabricated. These cuts are for offline ASR adaptation,
not for training causal streaming alignment or assistant responses.
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = pq.read_table(args.metadata, columns=["shard", "key", "text"]).to_pylist()
    transcripts = {(r["shard"], r["key"]): " ".join(r["text"].replace("+", "").split()) for r in rows}
    if len(transcripts) != len(rows):
        raise ValueError("Duplicate shard/key entries in metadata")
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "validation": []}
    counts = {"no_transcript": 0, "duration": 0, "token_capacity": 0}
    for shard in sorted(args.audio_dir.glob("*.tar")):
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".mp3"):
                    continue
                key = Path(member.name).stem
                text = transcripts.get((shard.stem, key))
                if not text:
                    counts["no_transcript"] += 1
                    continue
                audio = archive.extractfile(member).read()
                info = sf.info(io.BytesIO(audio))
                if info.channels != 1:
                    raise ValueError(f"Expected mono audio: {shard.name}/{key}")
                duration = info.frames / info.samplerate
                if not 0.5 <= duration <= 20:
                    counts["duration"] += 1
                    continue
                # Upstream has one token slot per 80 ms; never silently truncate references.
                if len(tokenizer.encode(text, add_special_tokens=False)) + 1 > int(duration / 0.08):
                    counts["token_capacity"] += 1
                    continue
                file_path = args.output / "audio" / shard.stem / f"{key}.mp3"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(audio)
                recording = Recording(
                    id=f"{shard.stem}-{key}",
                    sources=[AudioSource(type="file", channels=[0], source=str(file_path.resolve()))],
                    sampling_rate=info.samplerate,
                    num_samples=info.frames,
                    duration=duration,
                    channel_ids=[0],
                )
                cut = recording.to_cut()
                cut.supervisions = [
                    SupervisionSegment(
                        id=recording.id,
                        recording_id=recording.id,
                        start=0,
                        duration=duration,
                        channel=0,
                        text=text,
                        language="ru",
                        speaker="user",
                    )
                ]
                # Match upstream inference's initial PAD and leave room for EOS.
                cut = cut.resample(16000).pad(duration=duration + 0.08, direction="left")
                cut = cut.pad(duration=duration + 0.24)
                cut.custom = {"task": "asr"}
                # Identical normalized transcripts always remain in the same split.
                split = (
                    "validation" if int(hashlib.sha256(text.lower().encode()).hexdigest(), 16) % 100 == 0 else "train"
                )
                splits[split].append(cut)
        print(f"{shard.name}: train={len(splits['train'])}, validation={len(splits['validation'])}", flush=True)
    stats = {"excluded": counts, "alignment": "offline transcript; no word timestamps"}
    for name, cuts in splits.items():
        if not cuts:
            raise ValueError(f"Empty split: {name}")
        CutSet.from_cuts(cuts).to_file(args.output / f"{name}.jsonl.gz")
        stats[name] = {"samples": len(cuts), "hours": sum(c.duration - 0.24 for c in cuts) / 3600}
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
