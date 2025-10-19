# TODO - Next Session Checklist

## Immediate (Tonight/Tomorrow)

```markdown
- [ ] Wait for preprocessing to complete (~30-60 min)
- [ ] Verify 4 HDF5 files created (~12GB total)
- [ ] Start training: `./train_safe_tmux.sh`
- [ ] Monitor for 30 minutes (ensure no crashes)
- [ ] Let training run overnight
```

## Morning Check (Next Day)

```markdown
- [ ] Check training completed successfully
- [ ] Check NRMSE score (compare to baseline: 1.00)
- [ ] Verify model weights saved
- [ ] Check logs for any issues
```

## If Training Successful

```markdown
- [ ] Start Challenge 2 training
- [ ] Begin EEGNet implementation
- [ ] Research data augmentation methods
```

## If Any Issues

```markdown
- [ ] Check logs: `logs/training_comparison/*.log`
- [ ] Check memory usage patterns
- [ ] Review TRAINING_COMMANDS.md troubleshooting
- [ ] Post issue in relevant section
```

---

**Key Files:**
- README_SESSION_OCT18.md (overview)
- SESSION_SUMMARY_PART3_STATUS.md (current state)
- QUICK_COMMANDS.md (commands)

**Monitoring Commands:**
```bash
# Preprocessing
tail -f logs/preprocessing/cache_safe_*.log

# Training
tmux attach -t eeg_train_safe

# Memory
watch -n 5 'free -h'
```

