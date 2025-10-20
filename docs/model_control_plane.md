Model Control Plane Overview
============================

The **Model Control Plane (MCP)** is the governance and orchestration layer that
keeps our EEG 2025 pipelines consistent from experimentation through
production delivery. It coordinates four core responsibilities:

1. **Lifecycle Management** – standardising how we package, version, deploy,
   and retire EEG models. Each run receives a unique identifier that ties log
   output, checkpoints, dataset metadata, and evaluation reports together.
2. **Observability** – continuously collecting telemetry (training metrics,
   resource usage, ROCm fault reports) so incidents can be triaged quickly. The
   updated monitors (see `scripts/monitoring/`) now flag ROCm fallbacks and
   surface database health alongside GPU status.
3. **Policy Enforcement** – capturing compliance rules and guard-rails (e.g.
   checkpoint retention quotas, reproducibility requirements, evaluation gates).
4. **Access & Security** – centralising credentials, audit events, and role
   based permissions for training, evaluation, and model promotion tasks.

Key Components Implemented
--------------------------

- **Run Registry** – `data/metadata.db` tracks training runs, epochs, and
  checkpoints. Scripts in `scripts/training/` register runs up-front so every
  execution is visible to the control plane.
- **Telemetry Feed** – both `watch_training.sh` and
  `monitoring/enhanced_monitor.sh` stream status updates. They now detect the
  ROCm → CPU fallback path, which is recorded in the training logs for later
  analysis.
- **Operational Playbooks** – the revised training loop reports detailed
  timing, GPU diagnostics, and actionable ROCm guidance (including
  recommendations to check `PYTORCH_ROCM_ARCH` and
  `HSA_OVERRIDE_GFX_VERSION`). These messages feed into the MCP documentation.
- **Documentation Set** – this file (and the related README entries) anchor
  the concepts, whereas incident response guidance is cross-referenced in
  `docs/` and the training scripts themselves.

Operational Notes
-----------------

- GPU executions are attempted first. If ROCm raises
  `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, the control plane marks the run
  as degraded but keeps it alive by moving computation to CPU.
- Operators should consult the training log for the line
  `CPU fallback ready; continuing training without GPU acceleration.`. Both
  monitoring scripts surface the same warning.
- Before re-enabling GPU execution, validate ROCm environment variables:
  `PYTORCH_ROCM_ARCH`, `HSA_OVERRIDE_GFX_VERSION`, and `HIP_VISIBLE_DEVICES`.
- The Model Control Plane expects checkpoints to land in
  `checkpoints/challenge2_r1r2/` and for summary metrics to be registered via
  `log_epoch`. Downstream evaluation jobs read this data when promoting models
  to inference endpoints.

Next Steps
----------

1. Expand the MCP database schema to record fallback events explicitly
   (timestamp, reason, remedial action).
2. Integrate automated notification hooks (Slack / email) for future ROCm
   failures.
3. Attach compliance metadata (data slice provenance, reproducibility hashes)
   to the run registry so promotion decisions remain auditable.

For a high-level summary, see `README.md`. For operational runbooks, refer to
`docs/` alongside the monitoring scripts.
