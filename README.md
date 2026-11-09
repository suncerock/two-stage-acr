# Two-Stage ACR

Inference and example training code accompanying the paper "Revisiting Two-Stage Audio Chord Recognition in the Deep Learning Era: Leveraging Multi-Pitch Estimation and Symbolic Pretraining", in ISMIR 2026.

This repo is kept for reproducibility and research purposes. For a general-purpose, best-performing harmony recognition model, refer to [this repo](https://github.com/suncerock/automatic-harmony-analyzer), which contains also a demo page and will be actively maintained.

## Installation

The project uses Python 3.10. Install the dependencies in your environment with:

```bash
pip install -r requirements.txt
```

## Supported checkpoints

The repo supports inference with the four chord-model combinations:

- `crnn` + `classical`
- `crnn` + `pop`
- `conformer` + `classical`
- `conformer` + `pop`

The MPE frontend checkpoint is shared by all four models.

The checkpoints are hosted on Zenodo and can be downloaded via the following command:
```bash
for n in "pretrained_mpe" "pretrained_chord" "crnn_pop" "conformer_pop" "crnn_classical" "conformer_classical"; do
    wget "https://zenodo.org/record/21644912/files/$n.ckpt" -P ckpt/
done
```
Or download them manually and put them in the `ckpt/` folder.

## Inference

Run inference on one audio file:

```bash
python scripts/inference.py --audio-path /path/to/audio.wav --output-path /path/to/output/prefix \
  --backbone-type crnn --model-type classical [--soft-output]
```

The script writes:

- `/path/to/output/prefix.lab`
- `/path/to/output/prefix.npz` (if --soft-output is set)

## Notes

- `--model-type` selects `classical` or `pop`.
- `--backbone-type` selects `crnn` or `conformer`.
- The HCQT frontend is computed inside the inference script from raw audio.
- `.lab` file is the MIREX-style files with start time, end time, and chord symbols for each predicted segments
- `--soft-output` is false by default. If set to true, save a `.npz` file that contains keys `root`, `bass`, and `pc` storing frame-wise predictions (shape `12 x T`, unnormalized)
- For `crnn`, the `.npz` file also contains `chord` with frame-wise chord logits (shape `192 x T`, unnormalized, 12 * 16 qualities can be found in [src/data/utils.py](src/data/utils.py))
- Does not include `N` (no-chord) for now, coming soon...

## Datasets
The following datasets are used for training the chord recognition module and therefore should not serve as test data:
- Classical: SWD, BPSD, BSQD (Suggested test dataset: RUBATO, ...)
- Pop: Billboard, Isophonics (Suggested test dataset: RWC-Pop, USPop, ...)

## Training

An example training (fine-tuning) config is at [configs/mpe_chord_f_crnn_classical.yaml](configs/mpe_chord_f_crnn_classical.yaml).

Run it with:

```bash
python -m src.train --config configs/mpe_chord_f_crnn_classical.yaml
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ding2026twostageacr,
  author    = {Yiwei Ding and Yannik Venohr and Sebastian Strahl and Meinard M\"uller and Christof Wei{\ss}},
  title     = {Revisiting Two-Stage Audio Chord Recognition in the Deep Learning Era: Leveraging Multi-Pitch Estimation and Symbolic Pretraining},
  booktitle = {Proceedings the International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  address   = {Abu Dhabi, UAE},
}
```

## License

Code: MIT License

Pretrained weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — for research and academic use only, not for commercial use.
