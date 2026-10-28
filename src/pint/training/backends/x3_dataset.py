
import csv
import colorsys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Fixed saturation and value (brightness) for all colors
FIXED_SATURATION = 0.9
FIXED_VALUE = 0.9
TILE_SIZE = 3  # 3x3 pixel grid

class ColorGridDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        grid_size: int = 3,          # <-- NEW
        fixed_s: float = 0.9,
        fixed_v: float = 0.9,
        label_start: int | None = None,  # <-- optional override
    ):
        self.csv_path = csv_path
        self.grid_size = int(grid_size)
        self.num_tiles = self.grid_size * self.grid_size
        self.fixed_s = fixed_s
        self.fixed_v = fixed_v

        # columns: [0]=id, [1..num_tiles]=tiles, [label_start..]=labels
        self.label_start = (1 + self.num_tiles) if label_start is None else int(label_start)

        self.rows: List[List[str]] = []
        self.vocab = {}
        self._load()

    def _load(self):
        with open(self.csv_path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].lower() == "id":
                    continue
                self.rows.append(row)

        labels = set()
        for row in self.rows:
            for label in row[self.label_start:]:
                if label:
                    labels.add(label)
        labels = sorted(labels)
        self.vocab = {lab: i for i, lab in enumerate(labels)}

    def __len__(self):
        return len(self.rows)

    def _hues_from_row(self, row: List[str]) -> List[float]:
        hues = []
        # read exactly num_tiles hues from columns 1..num_tiles
        for i in range(1, 1 + self.num_tiles):
            val = row[i]
            try:
                hue = float(val)
            except Exception:
                hue = float(val.strip('"'))
            hues.append(hue)
        return hues

    def _image_from_hues(self, hues: List[float]) -> np.ndarray:
        # create H,W,C then transpose to C,H,W
        H = W = self.grid_size
        img = np.zeros((H, W, 3), dtype=np.float32)
        k = 0
        for i in range(H):
            for j in range(W):
                h = hues[k]
                r, g, b = colorsys.hsv_to_rgb(h, self.fixed_s, self.fixed_v)
                img[i, j, :] = (r, g, b)
                k += 1
        return np.transpose(img, (2, 0, 1))  # C,H,W

    def _cond_vector_from_row(self, row: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for label in row[self.label_start:]:
            if label and label in self.vocab:
                vec[self.vocab[label]] = 1.0
        return vec

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        hues = self._hues_from_row(row)
        img = self._image_from_hues(hues)
        cond = self._cond_vector_from_row(row)
        return torch.from_numpy(img).float(), torch.from_numpy(cond).float()


def quick_load(csv_path: str) -> ColorGridDataset:
	"""Helper to build dataset and return it (useful in scripts)."""
	return ColorGridDataset(csv_path)


if __name__ == '__main__':
	import sys
	if len(sys.argv) < 2:
		print('Usage: python3 dataset_loader.py generated_database/<seed>_<n>.csv')
		raise SystemExit(1)
	path = sys.argv[1]
	ds = quick_load(path)
	print(f'Loaded {len(ds)} samples, vocab size {len(ds.vocab)}')
	# show a small example
	img, cond = ds[0]
	print('Image tensor shape:', img.shape)
	print('Cond vector sum (num active labels):', int(cond.sum().item()))


