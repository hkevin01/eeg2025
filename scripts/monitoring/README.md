# Monitoring Scripts

Scripts for monitoring training, cache creation, and system status.

## Monitoring Scripts

- `dashboard.sh` - Comprehensive dashboard
  - Training status
  - Cache status
  - System resources
  - Database metrics

- `check_training_status.sh` - Quick training check
  - Current epoch
  - Latest metrics
  - Training time

- `advanced_monitor.sh` - Advanced monitoring
  - Detailed metrics
  - Resource usage
  - Performance stats

- `monitor_advanced.sh` - Alternative advanced monitor

- `manage_watchdog.sh` - Manage watchdog processes
  - Start/stop watchdog
  - Check watchdog status
  - View watchdog logs

## Usage

**View dashboard:**
```bash
bash scripts/monitoring/dashboard.sh
```

**Quick status:**
```bash
bash scripts/monitoring/check_training_status.sh
```

**Watch training:**
```bash
watch -n 5 bash scripts/monitoring/check_training_status.sh
```

**Database queries:**
```bash
sqlite3 data/metadata.db "SELECT * FROM training_runs;"
sqlite3 data/metadata.db "SELECT * FROM epoch_history ORDER BY epoch DESC LIMIT 10;"
```

---
*Organized: October 19, 2025*
