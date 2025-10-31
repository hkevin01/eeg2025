import re

with open('training_ensemble_fixed.log', 'r') as f:
    content = f.read()

# Split by seed
seeds = [42, 123, 456, 789, 999]
best_val_losses = {}

for seed in seeds:
    pattern = rf'🌱 Training Model with Seed {seed}.*?🏆 Best Val NRMSE for Seed {seed}'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        seed_section = match.group(0)
        # Find all "New best" lines with Val Loss
        best_lines = re.findall(r'Val Loss=([\d.]+).*?✅ New best', seed_section)
        if best_lines:
            best_val_loss = float(best_lines[-1])  # Last "New best" is the best
            best_val_losses[seed] = best_val_loss

print("�� Best Val Loss (MSE) for each seed:")
print("="*50)
for seed in seeds:
    if seed in best_val_losses:
        print(f"  Seed {seed:3d}: Val Loss (MSE) = {best_val_losses[seed]:.6f}")

if best_val_losses:
    avg_val_loss = sum(best_val_losses.values()) / len(best_val_losses)
    print(f"\n📈 Average Val Loss (MSE): {avg_val_loss:.6f}")
    print(f"🎯 V8 Val Loss (MSE):      0.079314")
    print(f"📊 Ensemble vs V8:         {((avg_val_loss/0.079314 - 1)*100):+.2f}%")
    if avg_val_loss < 0.079314:
        print("✅ Ensemble is BETTER!")
    else:
        print("❌ Ensemble is WORSE")
