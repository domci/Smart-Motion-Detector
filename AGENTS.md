## Supermemory

- Supermemory keeps durable decisions and lessons for this repository.
- Never save secrets. Use `<private>...</private>` for sensitive text.
- Run `$supermemory-status` when memory checks are needed.

<!-- ci-wait:start -->
## CI wait time

- Waiting on CI is work time. While a run is in flight, improve the pipeline: cut slow or redundant jobs and steps, cache dependencies, parallelize or shard, run only what the changed paths need, and drop checks that hooks or another job already own.
- Use the same wait for the `test-audit` skill (`/test-audit`): remove tests that re-assert source, duplicate stronger proof, or keep test-only seams alive; keep tests that guard shipped contracts.
- Ship CI and test cleanup as its own small commit or PR, separate from the feature change. Never weaken a release gate, security check, or required status just to make CI faster.
- Note the run's wall-clock time before and after in the PR or commit message.
<!-- ci-wait:end -->

## Disk hygiene

- Use the shared package caches on `server` (`/srv/dom/cache/bun`, `/srv/dom/cache/npm`). Never point `BUN_INSTALL_CACHE_DIR`, `npm_config_cache` or a `--cache-dir` at a scratchpad or temp folder; each private copy costs about 2 GB. If the shared cache looks broken, run `bun pm cache rm` instead of copying it.
- Never keep backups, database dumps, snapshots or release tarballs in a scratchpad, `/tmp` or `/srv/dom/tmp`; scratchpads are deleted after 7 days. Write them to `/media/external/backups/<project>/`.
- When the task is done, delete the large temp output you made: repro checkouts, build copies, downloaded data.
<!-- disk-hygiene:end -->
