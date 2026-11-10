import csv

with open("outputs/method_2_synthetic/filter_log.csv") as f:
    log = list(csv.DictReader(f))
rejected = [r for r in log if r["reason"] == "paved_surface"]

print(f"Paved surface rejections: {len(rejected)}")
print("First 10 rejected:")
for r in rejected[:10]:
    orig = float(r["paved_frac_orig"])
    fake = float(r["paved_frac_fake"])
    delta = abs(orig - fake)
    print(f"  source={r['source'][:30]} orig={orig:.3f} fake={fake:.3f} delta={delta:.3f}")

# Summary stats
deltas = [abs(float(r["paved_frac_orig"]) - float(r["paved_frac_fake"])) for r in rejected]
print()
print("Delta distribution:")
print(f"  min:  {min(deltas):.3f}")
print(f"  max:  {max(deltas):.3f}")
print(f"  mean: {sum(deltas) / len(deltas):.3f}")

# How many rejected have fake == 0.000 or fake == 1.000 (model failure)
fake_zeros = sum(1 for r in rejected if float(r["paved_frac_fake"]) < 0.01)
fake_ones = sum(1 for r in rejected if float(r["paved_frac_fake"]) > 0.99)
print()
print(f"fake_frac < 0.01: {fake_zeros}")
print(f"fake_frac > 0.99: {fake_ones}")
