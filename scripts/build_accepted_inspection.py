import csv
from pathlib import Path
from PIL import Image, ImageDraw

log = list(csv.DictReader(open('outputs/method_2_synthetic/filter_log.csv')))
accepted = [r for r in log if r['kept'] == '1']

out_dir = Path('outputs/method_2_synthetic/accepted_inspection')
out_dir.mkdir(parents=True, exist_ok=True)

source_root = Path(r'C:\Users\Janak Mallik\ml-data\bdd100k\images\100k\train')

# Sample 20 evenly spaced from accepted
step = max(1, len(accepted) // 20)
sample = accepted[::step][:20]

for i, r in enumerate(sample, 1):
    src_name = r['source']
    gen_name = r['output']
    
    src_path = None
    for p in source_root.rglob(src_name):
        src_path = p
        break
    
    if not src_path:
        continue
    
    src = Image.open(src_path).convert('RGB').resize((512, 288))
    gen = Image.open(Path('outputs/method_2_synthetic') / gen_name).convert('RGB').resize((512, 288))
    
    canvas = Image.new('RGB', (1024, 288 + 50), (30, 30, 30))
    canvas.paste(src, (0, 50))
    canvas.paste(gen, (512, 50))
    draw = ImageDraw.Draw(canvas)
    
    info = f'clip_n={r["clip_night"]} lum={r["luminance"]} ssim={r["ssim_diag"]}'
    draw.text((10, 15), f'SOURCE  {src_name[:30]}', fill=(255, 255, 255))
    draw.text((522, 15), f'ACCEPTED  {info}', fill=(255, 255, 255))
    
    canvas.save(out_dir / f'accepted_{i:02d}.jpg')

print(f'Saved 20 accepted comparisons to {out_dir}')
