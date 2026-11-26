"""Download the paired Base Streaming configuration and weights, then verify."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['hf', 'modelscope'], default='hf')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/base-streaming'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'model_weights.pt']
    if args.source == 'hf':
        from huggingface_hub import hf_hub_download
        for filename in files:
            hf_hub_download('robbyant/lingbot-map-v2-base-streaming', filename, local_dir=args.output)
    else:
        from modelscope import snapshot_download
        snapshot_download('Robbyant/lingbot-map-v2', local_dir=str(args.output), allow_patterns=files)
    config = json.loads((args.output / 'config.json').read_text())
    digest = hashlib.sha256()
    with (args.output / 'model_weights.pt').open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != config['sha256']:
        raise RuntimeError('Checkpoint SHA-256 mismatch; do not load this download.')
    print('Verified:', args.output.resolve())
    print('Pass --checkpoint <output>/model_weights.pt --config <output>/config.json to infer.py or demo.py.')


if __name__ == '__main__':
    main()
