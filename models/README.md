# Model checkpoint files

Each of the model files come with different licenses, so not uploading them here.

## File → OpenMM-ML potential name mapping

| File                              | OpenMM-ML `potential_name`             | `createSystem` kwarg |
|-----------------------------------|----------------------------------------|----------------------|
| `aceff_v2.0.ckpt`                | `aceff-2.0`                            | `ckpt_path`          |
| `MACE-OFF23_small.model`         | `mace-off23-small`                     | `modelPath`          |
| `MACE-OFF23_medium.model`        | `mace-off23-medium`                    | `modelPath`          |
| `MACE-OFF23_large.model`         | `mace-off23-large`                     | `modelPath`          |
| `MACE-OFF23b_medium.model`       | `mace-off23-medium` (alternate build)  | `modelPath`          |
| `MACE-OFF24_medium.model`        | `mace-off24-medium`                    | `modelPath`          |
| `fennix-bio1M.fnx`               | `fennix-bio1-medium`                   | `modelPath`          |
| `fennix-bio1M-finetuneIons.fnx`  | `fennix-bio1-medium-finetune-ions`     | `modelPath`          |
| `fennix-bio1S.fnx`               | `fennix-bio1-small`                    | `modelPath`          |
| `fennix-bio1S-finetuneIons.fnx`  | `fennix-bio1-small-finetune-ions`      | `modelPath`          |

## Sources

- **AceFF-2.0** checkpoint: <https://huggingface.co/Acellera/AceFF-2.0/resolve/main/aceff_v2.0.ckpt>
- **MACE-OFF** models: <https://github.com/ACEsuit/mace-off>
- **FeNNIx** models: <https://github.com/FeNNol-tools/FeNNol-PMC>
 
