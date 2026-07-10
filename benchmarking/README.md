# Benchmarking

`run_benchmark.py` scores the full `CrossModalConsistencyPipeline` against a
labeled fixture set and reports precision/recall/F1 per attack category.

## Using the built-in synthetic fixtures (default)

```bash
python -m benchmarking.run_benchmark --n-per-category 10
```

`attack_fixtures.py` procedurally generates four categories of labeled image
(+ caption) pairs:

| Category | How it's constructed | Meant to validate |
|---|---|---|
| `clean` | Smooth procedural gradient + light texture, plausible EXIF | No false positives on ordinary images |
| `adversarial_perturbation` | Clean image + high-frequency Gaussian noise | Noise-residual heuristic / adversarial-robustness path |
| `metadata_spoofed` | Clean pixels, EXIF `Software` tag set to an editor + inconsistent modify/capture dates | EXIF rule engine |
| `manipulated_splice` | A patch from a second image pasted in, then double-JPEG-compressed | Error-level analysis (ELA) |

These are **synthetic stand-ins**, not real photographs or real deepfakes.
They exist so the fusion/harness logic has something concrete to run against
in CI without bundling a multi-GB dataset. Treat the accuracy numbers in
`report.md` as "the pipeline plumbing works end-to-end," not "this detects
real-world deepfakes at X% accuracy."

## Using a real dataset

To get meaningful accuracy numbers, replace `build_fixture_set()` with a
loader over a real labeled dataset, keeping the same `Fixture` shape:

```python
from benchmarking.attack_fixtures import Fixture

def load_real_fixtures(dataset_dir: str) -> list[Fixture]:
    ...  # read images + labels from FaceForensics++ / DFDC / CASIA v2 / your own set
```

Then pass your loader's output into `run()` in `run_benchmark.py` in place of
`build_fixture_set(...)`. Popular public options:

- **FaceForensics++** — face-swap / reenactment deepfakes.
- **DFDC (Deepfake Detection Challenge)** — large-scale, diverse deepfakes.
- **CASIA v2** — classic splicing/copy-move manipulation benchmark, good for
  stress-testing the ELA path specifically.

Note these datasets have their own licenses/access requirements — check
before redistributing anything derived from them.
