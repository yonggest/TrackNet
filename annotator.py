"""
Tennis Ball Annotator
Usage: python annotator.py --clip_dir <path_to_clip_folder>

Controls:
  Left click       : mark ball position
  Right click      : mark as not visible (visibility=0)
  A / Left arrow   : previous frame
  D / Right arrow  : next frame
  1 / 2 / 3        : set visibility (1=clear, 2=blurry, 3=barely visible)
  0                : toggle not visible
  S                : save Label.csv
  Q / ESC          : quit (auto-saves)
"""

import cv2
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

WINDOW = 'Tennis Ball Annotator'
VIS_COLORS = {0: (100, 100, 100), 1: (0, 255, 0), 2: (0, 200, 255), 3: (0, 100, 255)}
VIS_LABELS = {0: 'Not Visible', 1: 'Visible (1)', 2: 'Blurry (2)', 3: 'Barely (3)'}


class Annotator:
    def __init__(self, clip_dir):
        self.clip_dir = Path(clip_dir)
        self.images = sorted([
            f for f in os.listdir(clip_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not self.images:
            raise ValueError(f'No images found in {clip_dir}')

        self.idx = 0
        self.annotations = {}  # filename -> {visibility, x, y, status}
        self.csv_path = self.clip_dir / 'Label.csv'
        self.load_csv()

    def load_csv(self):
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                self.annotations[row['file name']] = {
                    'visibility': int(row['visibility']),
                    'x': row['x-coordinate'] if not pd.isna(row['x-coordinate']) else None,
                    'y': row['y-coordinate'] if not pd.isna(row['y-coordinate']) else None,
                    'status': int(row['status']) if not pd.isna(row['status']) else 0,
                }
            print(f'Loaded {len(self.annotations)} annotations from {self.csv_path}')

    def save_csv(self):
        rows = []
        for fname in self.images:
            ann = self.annotations.get(fname)
            if ann:
                rows.append({
                    'file name': fname,
                    'visibility': ann['visibility'],
                    'x-coordinate': ann['x'] if ann['visibility'] != 0 else '',
                    'y-coordinate': ann['y'] if ann['visibility'] != 0 else '',
                    'status': ann['status'],
                })
            else:
                rows.append({
                    'file name': fname,
                    'visibility': 0,
                    'x-coordinate': '',
                    'y-coordinate': '',
                    'status': 0,
                })
        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
        print(f'Saved {len(rows)} rows to {self.csv_path}')

    def current_file(self):
        return self.images[self.idx]

    def draw_frame(self):
        img_path = self.clip_dir / self.current_file()
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, 'Cannot load image', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ann = self.annotations.get(self.current_file())
        vis = ann['visibility'] if ann else -1
        color = VIS_COLORS.get(vis, (200, 200, 200))

        # draw ball marker
        if ann and ann['visibility'] != 0 and ann['x'] is not None:
            x, y = int(ann['x']), int(ann['y'])
            cv2.circle(img, (x, y), 12, color, 2)
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 20, 2)

        # HUD
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        annotated = sum(1 for f in self.images if f in self.annotations)
        status_str = VIS_LABELS.get(vis, 'Unannotated')
        info = (f'  [{self.idx+1}/{len(self.images)}]  {self.current_file()}'
                f'  |  {status_str}'
                f'  |  Annotated: {annotated}/{len(self.images)}'
                f'  |  S=Save  A/D=Prev/Next  0-3=Visibility  Q=Quit')
        cv2.putText(img, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # visibility legend
        for i, (v, label) in enumerate(VIS_LABELS.items()):
            c = VIS_COLORS[v]
            marker = '>' if v == vis else ' '
            cv2.putText(img, f'{marker}{v}:{label}', (w - 200, 60 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        return img

    def on_mouse(self, event, x, y, flags, param):
        fname = self.current_file()
        if event == cv2.EVENT_LBUTTONDOWN:
            vis = self.annotations.get(fname, {}).get('visibility', 1)
            if vis == 0:
                vis = 1
            self.annotations[fname] = {'visibility': vis, 'x': x, 'y': y, 'status': 0}
            self.refresh()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.annotations[fname] = {'visibility': 0, 'x': None, 'y': None, 'status': 0}
            self.refresh()

    def refresh(self):
        cv2.imshow(WINDOW, self.draw_frame())

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 740)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.refresh()

        while True:
            key = cv2.waitKey(0) & 0xFF
            fname = self.current_file()

            if key in (ord('q'), 27):  # Q or ESC
                self.save_csv()
                break
            elif key in (ord('d'), 83):  # D or right arrow
                self.idx = min(self.idx + 1, len(self.images) - 1)
            elif key in (ord('a'), 81):  # A or left arrow
                self.idx = max(self.idx - 1, 0)
            elif key == ord('s'):
                self.save_csv()
            elif key in (ord('0'), ord('1'), ord('2'), ord('3')):
                v = int(chr(key))
                ann = self.annotations.get(fname, {'x': None, 'y': None, 'status': 0})
                ann['visibility'] = v
                if v == 0:
                    ann['x'] = None
                    ann['y'] = None
                self.annotations[fname] = ann

            self.refresh()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tennis Ball Annotator')
    parser.add_argument('--clip_dir', type=str, required=True,
                        help='Path to folder containing images to annotate')
    args = parser.parse_args()

    annotator = Annotator(args.clip_dir)
    annotator.run()
