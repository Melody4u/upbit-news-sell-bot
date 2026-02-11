# Rollback Policy

## Purpose
Prevent prolonged degradation during live/paper operations.

## Predefined rollback triggers
Rollback immediately if any trigger is met during validation window:

1) MDD worsens by >= 1.0%p vs baseline
2) PF drops below baseline by >= 0.15
3) Pending timeout count increases >= 50%
4) Order/position state inconsistency detected

## Rollback action
1) Revert to previous stable commit
2) Restore previous .env preset
3) Mark experiment as failed with root-cause notes

## Baseline discipline
- Always keep one "known-good" tag/commit
- Compare only against same market and similar session conditions

## Communication
- Log rollback reason in experiment report
- If alerts are enabled, send rollback event summary
