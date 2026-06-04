# data/

This directory holds the VisDrone dataset and is **gitignored** (it is
several GB). Only this README is tracked.

## Layout (after running the data pipeline)

```
data/
├── raw/                          # downloaded + extracted VisDrone archives
│   ├── VisDrone2019-DET-train/
│   ├── VisDrone2019-DET-val/
│   └── VisDrone2019-MOT-val/
└── yolo/
    └── VisDrone-DET/             # converted YOLO-format dataset
        ├── images/{train,val,test}/
        └── labels/{train,val,test}/
```

## How to populate

```bash
# Download (resumable, cached) + convert to YOLO format + split + validate
python -m src.data.download_visdrone --config configs/paths.yaml
python -m src.data.convert_visdrone  --config configs/paths.yaml
python -m src.data.validate_labels   --config configs/paths.yaml
```

See the repository README for details.
