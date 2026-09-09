# Supermemory

## Purpose

Supermemory keeps durable project decisions and operating lessons. It is context, not the source of truth. Check code, tests, and live state before acting.

## Scope

Codex uses one container per Git remote. Clones and worktrees of this repository share memory. Other repositories stay separate.

## Use

Recall and capture are automatic. Run `$supermemory-status` to check the connection and current repository container.

Save only decisions, constraints, and repeatable lessons. Do not save secrets. Wrap sensitive text in `<private>...</private>`.

## Hermes

Hermes uses the same homeserver service. For a repository-specific Hermes task, use this repository container. Add its tag to the Hermes custom-container allowlist first.
