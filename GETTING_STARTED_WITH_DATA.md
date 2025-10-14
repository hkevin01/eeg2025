# 🚀 Getting Started With Real Data

**You Are Here**: Ready to get real EEG data and start training!  
**Time Required**: 2-4 hours  
**Difficulty**: Easy (step-by-step instructions provided)

---

## 🎯 What You Need To Do

The project has great code but **NO REAL DATA YET**. Here's how to fix that:

### Option 1: Quick Start (Recommended) ⚡

```bash
# 1. Install data tools (2 minutes)
pip install mne mne-bids boto3 requests tqdm

# 2. Create directories (10 seconds)
mkdir -p data/raw/hbn data/processed data/cache

# 3. Download 2 sample subjects (15 minutes)
python scripts/download_hbn_data.py --subjects 2 --verify

# 4. Test it works (2 minutes)
python scripts/verify_data_structure.py --data-dir data/raw/hbn
```

**Done!** You now have real data to work with.

---

### Option 2: Read The Full Guide First 📖

If you want detailed explanations:

```bash
# Read the comprehensive guide
cat docs/DATA_ACQUISITION_GUIDE.md
```

This includes:
- Why we need this data
- What the data contains
- How to troubleshoot issues
- Quality checks to run

---

## 📚 All Available Resources

We created **5 documents** to help you:

| Document | What It Does | When To Use |
|----------|--------------|-------------|
| **DATA_ACQUISITION_INDEX.md** | Master index of everything | Start here to orient yourself |
| **DATA_ACQUISITION_GUIDE.md** | Full tutorial with details | First time doing this |
| **QUICK_START_DATA_TODO.md** | Checklist format | You want speed |
| **WEEK_BY_WEEK_PLAN.md** | 4-week competition plan | Long-term planning |
| **DAILY_CHECKLIST.md** | Daily task template | Ongoing work |

All files are in the `docs/` folder.

---

## ✅ Simple 3-Step Process

### Step 1: Setup (15 min)
```bash
pip install mne mne-bids boto3 requests tqdm
mkdir -p data/raw/hbn data/processed data/cache
```

### Step 2: Download (15-30 min)
```bash
python scripts/download_hbn_data.py --subjects 2 --verify
```

### Step 3: Validate (5 min)
```bash
python scripts/verify_data_structure.py --data-dir data/raw/hbn
python -c "from src.dataio.hbn_dataset import HBNDataset; print('✓ Works!')"
```

---

## 🎓 What Happens Next

Once you have data:

1. **Train a baseline model** (Week 1-2)
   ```bash
   python src/training/train_cross_task.py --config configs/challenge1_baseline.yaml
   ```

2. **Optimize performance** (Week 2-3)
   - Add artifact detection
   - Tune hyperparameters
   - Try ensembles

3. **Submit to competition** (Week 4)
   ```bash
   python scripts/generate_submission.py --challenge challenge1
   ```

---

## 🆘 Need Help?

### Quick Checks
```bash
# Is everything installed?
python -c "import mne, boto3; print('✓ Ready')"

# Do I have data?
ls data/raw/hbn/sub-*/ses-*/eeg/

# Can I load data?
python -c "from src.dataio.hbn_dataset import HBNDataset"
```

### Troubleshooting
- **Download fails**: Check `docs/DATA_ACQUISITION_GUIDE.md` Section 6
- **Can't load data**: Verify BIDS structure with `scripts/verify_data_structure.py`
- **Out of memory**: Use `export MNE_USE_MEMMAP=1`

---

## 📊 Current Project Status

```
✅ CI/CD Pipeline: All tests passing
✅ Model Code: Advanced transformer architecture ready
✅ Training Scripts: Challenge 1 & 2 trainers implemented
✅ GPU Optimization: Triton kernels for speed
⚠️ Real Data: YOU ARE HERE - Need to download
⚠️ Testing: Need more integration tests
⚠️ Validation: Need to measure baseline performance
```

---

## 🎯 Your Mission

**Primary Goal**: Get 2 subjects downloaded and validated (2 hours)

**Secondary Goal**: Download 10 subjects for training (4 hours)

**Stretch Goal**: Download full dataset for best performance (overnight)

---

## 🚀 Ready? Start Here:

### For The Impatient (10 minutes to first data):
```bash
pip install mne mne-bids boto3 requests tqdm && \
mkdir -p data/raw/hbn && \
python scripts/download_hbn_data.py --subjects 1 --verify
```

### For The Thorough (Read first):
```bash
cat docs/DATA_ACQUISITION_INDEX.md
```

---

**Remember**: You can't train without data. This is THE critical first step! 🎯

**Questions?** Check `docs/DATA_ACQUISITION_INDEX.md` for the complete guide.

---

**Next Steps After Data**:
1. ✅ Get Data (THIS DOCUMENT)
2. 📝 Write Tests (`docs/WEEK_BY_WEEK_PLAN.md` Week 1)
3. 🎯 Train Baseline (`docs/WEEK_BY_WEEK_PLAN.md` Week 2)
4. 🚀 Optimize & Submit (`docs/WEEK_BY_WEEK_PLAN.md` Week 3-4)

**Let's go! 💪**
