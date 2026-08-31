# Task 18 subtask — Azure CLI + credentials setup on this machine

**Parent task:** `tasks/18-cloud-migration-of-experiments.md`
**Blocks:** every step in `tasks/18-cloud-migration-of-experiments_plan.md`
§3 (all of it requires an authenticated `az` session on this machine) —
this subtask must be done first.

## Why this is its own subtask

This machine has never talked to Azure before. Everything in task 18's
plan (`az storage account create`, `azcopy`, `az vm create`, etc.) requires
the Azure CLI to be installed *and* authenticated to the right account
first — a one-time local-machine setup step, distinct from the actual
experiment-migration work, and worth checking off separately so it isn't
silently assumed done partway through the real plan.

## 1. Current state (checked 2026-08-30)

- **Azure CLI: already installed.** `az --version` reports `azure-cli
  2.84.0`. No install step needed on this machine.
- **Authentication: not set up.** `az account show` fails with `ERROR:
  Please run 'az login' to setup account.` — this machine has no Azure
  session yet.

## 2. Steps to complete

1. **Authenticate**: `az login` (opens a browser) or, on a headless/SSH
   session with no local browser, `az login --use-device-code` (prints a
   URL + one-time code to enter on any other device's browser). This step
   requires interactive user action — it cannot be completed
   non-interactively.
2. **Verify the right subscription is active**: `az account show` after
   login; if the account has more than one subscription, `az account list
   --output table` then `az account set --subscription <name-or-id>` to
   pick the one to use for this work.
3. **Verify the account can actually create resources** (not just read):
   a harmless check like `az group list --output table` should succeed
   without an authorization error. If it fails, the account needs a role
   like Contributor on the target subscription — resolve with whoever
   administers the Azure subscription before starting the real plan.
4. **Optional convenience**: once a region is chosen for task 18's plan
   (the plan doc suggests East US, to match where NOAA's NODD mirror
   already lives, though this task doesn't use that mirror directly),
   `az configure --defaults location=<region>` saves having to pass
   `--location` on every later command.

## 3. Security note

`az login` stores tokens under `~/.azure` on this machine, **outside the
repo** — never commit that directory, any token, or a connection
string/SAS token into git. The Spot VM provisioned in task 18's plan will
need its *own*, separate way to reach Azure Blob Storage (e.g. a SAS token
or connection string generated for that purpose and passed to `azcopy` on
the VM) — it does not reuse this machine's `az login` session, which is
local-machine-only.

## 4. Conclusion

Done, with two real blockers found and resolved along the way (both worth
recording since they change the plan doc's assumptions):

1. **First account tried (`mmejiam@eafit.edu.co`, "Azure for Students
   Starter", university-managed tenant) is unusable for this task.**
   `Microsoft.Compute` is explicitly in the `DisallowedProvider` list at
   the tenant-policy level — not a quota limit, a hard block — and
   `Microsoft.Storage` is absent from the same allowed-provider list.
   Neither VMs nor Blob Storage can be created on this account, ever,
   under its current policy. Switched to a second, personal account.
2. **Second account (`mmejiam9206@outlook.com`, "Azure subscription 1")
   works, but is a Free Trial subscription with real quota limits.**
   `Microsoft.Compute`/`Microsoft.Storage` were `NotRegistered` (normal for
   an unused subscription, not a block) — registered successfully via `az
   provider register`. But default regional quota in `eastus` is **4 total
   vCPUs (`cores`) and only 3 for Spot/low-priority (`lowPriorityCores`)**
   — below the plan's original D8s_v5 (8 vCPU) target, spot or otherwise.
   Attempted a self-service quota increase (`az quota create`) — failed
   with `ResourceNotAvailableForOffer`. Root cause: this subscription's
   `quotaId` is `FreeTrial_2014-09-01` with `spendingLimit: On` — Azure
   blocks quota increases entirely for Free Trial subscriptions until
   upgraded to Pay-As-You-Go (requires adding a payment method; doesn't
   charge anything while free credit remains, it just unlocks the quota
   mechanism). Upgrading is a portal action only the account owner can do.
3. **Resolution (2026-08-31): proceed now at the current quota rather than
   wait on a PAYG upgrade.** Target VM changed from D8s_v5 spot to
   **D4s_v5 on-demand (4 vCPU/16GB)** — fits the existing 4-vCPU regular
   quota with no waiting. A D2s_v5 spot (2 vCPU, fits the 3-vCPU spot cap)
   was also available but rejected: local runs this session used ~6-7
   active BLAS threads, so 2 vCPUs would be a step *down* from what's
   already available locally, defeating the point of migrating. See
   `tasks/18-cloud-migration-of-experiments_plan.md` §3 for the updated
   VM spec. Revisit a PAYG upgrade + real quota increase later if 4 vCPUs
   proves insufficient.
