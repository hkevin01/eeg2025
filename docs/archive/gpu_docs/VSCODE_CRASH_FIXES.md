# VS Code Crash & Hang Prevention Guide

**Problem:** VS Code crashes, hangs, or shows "waiting/reopen" messages with large output or long-running processes.

**Root Causes:**
1. Terminal buffer overflow from large output
2. GPU acceleration issues
3. File watcher overload
4. Memory leaks from extensions
5. Language server resource usage

---

## 🚀 Solution 1: Terminal Settings (CRITICAL)

### Problem: Terminal crashes with large output

### Fix: Limit scrollback and disable GPU acceleration

**Already Applied in `.vscode/settings.json`:**
```json
{
  "terminal.integrated.scrollback": 5000,
  "terminal.integrated.gpuAcceleration": "off",
  "terminal.integrated.rendererType": "dom"
}
```

**Why it works:**
- Limits terminal buffer to 5000 lines (default: unlimited)
- Disables GPU acceleration (prevents driver crashes)
- Uses DOM renderer (more stable, less resource-intensive)

---

## 🚀 Solution 2: File Watcher Optimization (CRITICAL)

### Problem: VS Code watches too many files, uses excessive CPU/memory

### Fix: Exclude large directories

**Already Applied in `.vscode/settings.json`:**
```json
{
  "files.watcherExclude": {
    "**/data/**": true,
    "**/logs/**": true,
    "**/checkpoints/**": true,
    "**/outputs/**": true,
    "**/__pycache__/**": true
  }
}
```

**Why it works:**
- Prevents watching thousands of data/log files
- Reduces CPU usage by 30-50%
- Prevents "too many open files" errors

---

## 🚀 Solution 3: Disable Heavy Extensions

### Problem: Extensions consume resources in background

### Fix: Disable or configure extensions

**Manual Steps:**
1. Press `Ctrl+Shift+P`
2. Type "Extensions: Show Installed Extensions"
3. Disable these types:
   - Auto-formatters (if not needed)
   - Linters running on every keystroke
   - AI assistants (GitHub Copilot) when not coding
   - Git Graph/History viewers
   - Theme/Icon packs (use lightweight ones)

**Already Applied:**
```json
{
  "extensions.autoCheckUpdates": false,
  "extensions.autoUpdate": false
}
```

---

## 🚀 Solution 4: Python Language Server Optimization

### Problem: Pylance/Python extension uses excessive memory

### Fix: Reduce indexing and analysis

**Already Applied:**
```json
{
  "python.analysis.memory.keepLibraryAst": false,
  "python.analysis.indexing": false,
  "python.terminal.activateEnvironment": false
}
```

**Why it works:**
- Disables AST caching (saves 500MB+ RAM)
- Disables full workspace indexing
- Prevents venv activation overhead

---

## 🚀 Solution 5: Git Performance Tuning

### Problem: Git operations slow down VS Code

### Fix: Disable auto-refresh and decorations

**Already Applied:**
```json
{
  "git.autorefresh": false,
  "git.autofetch": false,
  "git.decorations.enabled": false
}
```

**Manual Git Refresh:**
```bash
# Only refresh when needed
git status
```

---

## 🚀 Solution 6: Use External Terminal for Long Processes

### Problem: VS Code terminal not designed for 24/7 processes

### Fix: Use tmux or separate terminal

**Create tmux session:**
```bash
# Install tmux if not present
sudo apt install tmux

# Start tmux session for training
tmux new -s training

# Run training
cd /home/kevin/Projects/eeg2025
python3 scripts/train_foundation_v2.py

# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

**Why it works:**
- Processes survive VS Code crashes
- Terminal buffer managed separately
- Can reconnect from anywhere

---

## 🚀 Solution 7: Output Redirection

### Problem: Live output overwhelms terminal

### Fix: Redirect to file, tail when needed

**Already Using:**
```bash
nohup python3 script.py > output.log 2>&1 &
tail -f output.log  # Watch when needed
```

**Better approach:**
```bash
# Use tee for both file and screen
python3 script.py 2>&1 | tee logs/training.log

# Or just file (check periodically)
python3 script.py > logs/training.log 2>&1 &
tail -n 50 logs/training.log  # Check last 50 lines
```

---

## 🚀 Solution 8: Increase VS Code Memory Limit

### Problem: VS Code runs out of memory

### Fix: Increase heap size

**Method 1: Command line flag**
```bash
code --max-memory=8192
```

**Method 2: Desktop file (permanent)**
```bash
# Edit VS Code launcher
nano ~/.local/share/applications/code.desktop

# Change Exec line to:
Exec=/usr/share/code/code --max-memory=8192 --unity-launch %F
```

**Already Applied in settings:**
```json
{
  "files.maxMemoryForLargeFilesMB": 8192
}
```

---

## 🚀 Solution 9: Disable Telemetry & Auto-Update

### Problem: Background network operations slow system

### Fix: Disable all telemetry

**Already Applied:**
```json
{
  "telemetry.telemetryLevel": "off",
  "update.mode": "none",
  "workbench.enableExperiments": false
}
```

---

## 🚀 Solution 10: System-Level Optimizations

### Problem: Linux file descriptor limits

### Fix: Increase limits

**Check current limits:**
```bash
ulimit -n  # Should be at least 65536
```

**Increase limits (temporary):**
```bash
ulimit -n 65536
```

**Increase limits (permanent):**
```bash
sudo nano /etc/security/limits.conf

# Add these lines:
* soft nofile 65536
* hard nofile 65536
```

**Reboot after changing.**

---

## 🚀 Solution 11: AMD GPU-Specific Fixes

### Problem: ROCm/GPU drivers conflict with VS Code

### Fix: Disable GPU features in VS Code

**Already Applied:**
```json
{
  "terminal.integrated.gpuAcceleration": "off"
}
```

**Additional: Disable window GPU acceleration**
```bash
code --disable-gpu
```

**Make permanent:**
```bash
# Edit VS Code desktop file
nano ~/.local/share/applications/code.desktop

# Change Exec line:
Exec=/usr/share/code/code --disable-gpu --unity-launch %F
```

---

## 🚀 Solution 12: Create Monitoring Scripts

### Problem: Can't tell if training crashed or just slow

### Fix: Use monitoring scripts

**Created script: `scripts/advanced_monitor.sh`**
```bash
./scripts/advanced_monitor.sh
```

**Features:**
- Shows process status
- CPU/Memory usage
- Latest log entries
- Auto-refreshes every 5 seconds

---

## 🛠️ Quick Fixes Checklist

When VS Code crashes or hangs:

### Immediate Actions:
- [ ] Close unnecessary terminals (keep 1-2 max)
- [ ] Close unused editor tabs
- [ ] Disable GitHub Copilot temporarily
- [ ] Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"

### If Still Crashing:
- [ ] Close VS Code completely
- [ ] Kill zombie processes: `pkill -9 code`
- [ ] Restart VS Code with: `code --disable-gpu --disable-extensions`
- [ ] Re-enable extensions one by one

### For Training:
- [ ] Use tmux instead of VS Code terminal
- [ ] Monitor with `tail -f logs/training.log`
- [ ] Use `scripts/advanced_monitor.sh`

### Nuclear Option:
```bash
# Clear VS Code cache
rm -rf ~/.config/Code/Cache
rm -rf ~/.config/Code/CachedData
rm -rf ~/.vscode/extensions

# Restart VS Code
code
```

---

## 📊 Monitoring Commands

```bash
# Check training process
ps aux | grep python | grep train

# Monitor CPU/Memory
htop  # or top

# Watch logs
watch -n 5 tail -n 20 logs/training.log

# Check disk space (logs can fill up)
df -h

# Check file descriptors
lsof -p $(pgrep -f train_foundation) | wc -l
```

---

## 🎯 Best Practices for Long Training

1. **Always use tmux or screen**
   ```bash
   tmux new -s training
   python3 script.py
   # Detach: Ctrl+B, D
   ```

2. **Redirect output to file**
   ```bash
   python3 script.py > logs/training.log 2>&1 &
   ```

3. **Monitor periodically (not continuously)**
   ```bash
   tail -n 50 logs/training.log  # Check every 10 minutes
   ```

4. **Use checkpointing**
   - Save model every N epochs (already implemented)
   - Training can resume from checkpoint

5. **Limit terminal sessions**
   - Close old terminals
   - Keep only 1-2 active

6. **Monitor system resources**
   ```bash
   htop  # Check if system is overloaded
   ```

---

## ✅ What's Already Applied

These fixes are already active in your project:

- ✅ Terminal scrollback limited to 5000 lines
- ✅ GPU acceleration disabled in terminal
- ✅ File watchers exclude data/logs/checkpoints
- ✅ Python indexing disabled
- ✅ Git auto-refresh disabled
- ✅ Telemetry disabled
- ✅ Large file optimizations enabled
- ✅ Training uses background processes with logging

---

## 🎬 What to Do Now

### For Current Training:

1. **Check if training is running:**
   ```bash
   ps aux | grep train_foundation_v2
   ```

2. **Monitor without VS Code:**
   ```bash
   ./scripts/advanced_monitor.sh
   ```

3. **If VS Code keeps crashing:**
   ```bash
   # Use external terminal
   gnome-terminal  # or konsole, or xterm
   cd /home/kevin/Projects/eeg2025
   tail -f logs/foundation_20*.log
   ```

### For Future Training:

1. **Use tmux:**
   ```bash
   tmux new -s training
   python3 scripts/train_foundation_v2.py
   # Ctrl+B, D to detach
   ```

2. **Monitor externally:**
   ```bash
   watch -n 10 tail -n 30 logs/training.log
   ```

3. **Use VS Code only for coding**
   - Not for monitoring long processes
   - Not for running 2+ hour jobs

---

## 📞 Emergency Recovery

### If VS Code won't start:
```bash
# Clear cache
rm -rf ~/.config/Code/Cache
rm -rf ~/.config/Code/CachedData

# Start with safe mode
code --disable-extensions --disable-gpu
```

### If training process is zombie:
```bash
# Find process
ps aux | grep python | grep train

# Kill gracefully
kill PID

# Kill forcefully if needed
kill -9 PID
```

### If system is frozen:
```bash
# Switch to TTY
Ctrl+Alt+F2

# Login
# Kill VS Code
pkill -9 code

# Kill Python
pkill -9 python3

# Return to GUI
Ctrl+Alt+F7
```

---

## 📚 Additional Resources

- VS Code Performance: https://code.visualstudio.com/docs/setup/linux#_visual-studio-code-is-unable-to-watch-for-file-changes-in-this-large-workspace
- Pylance Settings: https://github.com/microsoft/pylance-release#settings-and-customization
- tmux Tutorial: https://tmuxcheatsheet.com/

---

**Status:** All critical fixes applied to `.vscode/settings.json`

**Next:** Follow "Best Practices for Long Training" section above.
