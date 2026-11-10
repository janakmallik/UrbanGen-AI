import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms

# Load filter log
with open("outputs/method_2_synthetic/filter_log.csv") as f:
    log = list(csv.DictReader(f))
rejected = [r for r in log if r['reason'] == 'paved_surface']
rejected_sorted = sorted(rejected, key=lambda r: abs(float(r['paved_frac_orig']) - float(r['paved_frac_fake'])), reverse=True)
top5 = rejected_sorted[:5]

# Load CycleGAN (must match training architecture)
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c),
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(128) for _ in range(4)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 7, 1, 3), nn.Tanh(),
        )
    def forward(self, x):
        return self.dec(self.res(self.enc(x)))

device = 'cuda'
ckpt = torch.load(r'outputs/method_2_resume/checkpoints/epoch_20.pth', map_location='cpu', weights_only=False)
G_AB = Generator().to(device)
G_AB.load_state_dict(ckpt['G_AB'], strict=True)
G_AB.eval()
del ckpt

transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Build comparison images
out_dir = Path('outputs/method_2_synthetic/rejected_inspection')
out_dir.mkdir(parents=True, exist_ok=True)

source_root = Path(r'C:\Users\Janak Mallik\ml-data\bdd100k\images\100k\train')

for i, r in enumerate(top5, 1):
    src_name = r['source']
    src_path = None
    for p in source_root.rglob(src_name):
        src_path = p
        break
    if not src_path:
        print(f'Missing source: {src_name}')
        continue

    src = Image.open(src_path).convert('RGB').resize((512, 288))
    tensor = transform(Image.open(src_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        fake = G_AB(tensor)
    gen = Image.fromarray(((fake.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    canvas = Image.new('RGB', (1024, 288 + 50), (30, 30, 30))
    canvas.paste(src, (0, 50))
    canvas.paste(gen, (512, 50))
    draw = ImageDraw.Draw(canvas)
    delta = abs(float(r['paved_frac_orig']) - float(r['paved_frac_fake']))
    info = f'orig={r["paved_frac_orig"]} fake={r["paved_frac_fake"]} delta={delta:.3f}'
    draw.text((10, 15), f'SOURCE (day)  {src_name[:30]}', fill=(255, 255, 255))
    draw.text((522, 15), f'GENERATED (night)  {info}', fill=(255, 255, 255))
    canvas.save(out_dir / f'rejected_{i:02d}.jpg')
    print(f'Saved rejected_{i:02d}.jpg')

print(f'\nDone. Check {out_dir}')
