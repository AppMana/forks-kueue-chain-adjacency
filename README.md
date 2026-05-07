# Kueue — chain-adjacency fork

[![Upstream](https://img.shields.io/badge/upstream-kubernetes--sigs%2Fkueue-blue)](https://github.com/kubernetes-sigs/kueue)

This is a fork of [`kubernetes-sigs/kueue`](https://github.com/kubernetes-sigs/kueue) that adds **rank-deterministic placement on 1-D ordered topologies** with optional **compaction** (relocating bound workloads to defragment a chain).

It exists to make NCCL ring collectives, NVLink chains, Slingshot dragonfly inter-group traffic, and other adjacency-sensitive fabrics run at full bandwidth on Kubernetes — the existing Kueue [TAS API (KEP-2724)](https://github.com/kubernetes-sigs/kueue/blob/main/keps/2724-topology-aware-scheduling/README.md) handles set-based placement well but doesn't pin rank N to physical position N or defragment. This fork does both.

Everything else is upstream Kueue.

---

## What's different

### 1. `Topology.spec.levels[].ordered` (new field)

Mark one level on a `Topology` as `ordered: true`. Domain values at that level must parse as non-negative integers; the integer order is taken to reflect physical adjacency.

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: Topology
metadata: {name: chain}
spec:
  levels:
  - nodeLabel: topology.example.com/chain-name
  - nodeLabel: topology.example.com/chain-index
    ordered: true
  - nodeLabel: kubernetes.io/hostname
```

When a workload requests `kueue.x-k8s.io/podset-slice-required-topology` at an ordered level, the scheduler:
- Allocates a **leftmost contiguous run** of indices (no fragmentation, no scattered placement).
- Maps **rank N → start+N** deterministically (the existing rank-aware ungater consumes the assignment in order; this fork makes the assignment honour the integer sort, not lex).

If no contiguous run of the required size exists, the workload stays Pending unless compaction is enabled (below).

### 2. `ClusterQueue.spec.preemption.maxEvictionsPerSchedulingPass` (new field)

The compaction budget — the number of bound workloads the scheduler may evict per scheduling pass to defragment an ordered chain and admit a pending workload.

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata: {name: chain}
spec:
  preemption:
    maxEvictionsPerSchedulingPass: 1   # nil/0 = never compact
  ...
```

Semantics:
- `nil` or `0` (default): never compact. First-fit only.
- `N > 0`: compact up to N bound workloads per pass.

The allocator is a 1-D arena allocator framed as a compacting GC. It picks the lowest-cost feasible plan, ranking victim sets by:

1. Fewest evictions.
2. Lex-max victim ordering (minimise the worst victim's priority — matches `pkg/scheduler/preemption/common/ordering.go:CandidatesOrdering`).
3. Smallest total relocation distance.
4. Leftmost placement (deterministic tiebreak).

Compaction never displaces a higher-priority workload to admit a lower-priority one. Older workloads are protected against newer at equal priority. Non-evictable bounds (`ordered: false` on the Topology level, or no `tas-ordered-evictable` annotation on the workload) act as pinned obstacles.

### 3. Workload-level evictability opt-in

```yaml
metadata:
  annotations:
    kueue.x-k8s.io/tas-ordered-evictable: "true"
```

Default is `false` (workloads are not relocated). Job/JobSet/LWS integrations that want different per-kind defaults can set this in their parent webhooks.

---

## Quick start

```bash
# Install the fork on an existing cluster (replaces upstream Kueue).
kubectl apply -k 'github.com/AppMana/forks-kueue-chain-adjacency/config/default?ref=chain-adjacency'

# Or build a local image and load into kind:
make image-build IMAGE_REGISTRY=harbor.appmana.com/appmana-shared PLATFORMS=linux/amd64
kind load docker-image harbor.appmana.com/appmana-shared/kueue:main
```

Define a Topology with one ordered level (see above), a ResourceFlavor pointing at it, a ClusterQueue with the budget set, and apply your workloads with the standard Kueue TAS annotations:

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: chain
  annotations:
    kueue.x-k8s.io/podset-required-topology: topology.example.com/chain-name
    kueue.x-k8s.io/podset-slice-required-topology: topology.example.com/chain-index
    kueue.x-k8s.io/podset-slice-size: "1"
    kueue.x-k8s.io/tas-ordered-evictable: "true"   # opt in to compaction
```

---

## How it works

| Layer | What | Where |
|---|---|---|
| API | `Ordered` on `TopologyLevel`, `MaxEvictionsPerSchedulingPass` on `ClusterQueuePreemption` | `apis/kueue/v1beta2/{topology,clusterqueue}_types.go` |
| Snapshot | `levelOrdered` on `TASFlavorSnapshot`; numeric sort at ordered levels; per-workload `boundOrderedAllocations` populated from the cache | `pkg/cache/scheduler/tas_flavor_snapshot.go`, `tas_flavor.go`, `tas_cache.go` |
| Allocator | Pure-Go compacting-GC 1-D allocator; pluggable `victimComparator` for plan-level cost (defaults to lex-max-priority + age) | `pkg/cache/scheduler/tas_ordered.go` |
| Dispatch | At ordered child levels, contiguous-run pick replaces the upstream capacity-first set pick; falls through to `findCompactionPlan` when budget > 0 | `pkg/cache/scheduler/tas_flavor_snapshot.go: findTopologyAssignment` |
| Result | `tasPodSetAssignmentResult.CompactionEvictions []workload.Reference` carries the eviction list | `pkg/cache/scheduler/tas_flavor_snapshot.go` |
| Scheduler | `Assignment.TASCompactionEvictions` is drained via `scheduler.issueTASCompaction` (same shape as `issueMigration`); pending workload requeues | `pkg/scheduler/{flavorassigner,scheduler}.go` |

`ordered: false` is the default, so existing TAS workloads — and the entire upstream Kueue test suite — are bit-for-bit unaffected.

---

## Tests

```bash
go test -race ./pkg/cache/scheduler/ ./pkg/scheduler/...
```

Notable coverage in this fork:

- `pkg/cache/scheduler/tas_ordered_test.go` — 36+ pure-Go allocator scenarios. Memory-allocator-style fragmentation patterns (checkerboard, worst-case), budget knob spectrum (0 → ∞), priority and age tiebreakers, pluggable comparator, LWS-vs-JobSet evictability semantics on a 12-node chain.
- `pkg/cache/scheduler/tas_ordered_snapshot_test.go` — numeric ordering at ordered levels; `Ordered=false` regression guard verifying upstream lex behaviour is preserved.
- `pkg/cache/scheduler/tas_ordered_dispatch_test.go` — end-to-end through `FindTopologyAssignmentsForFlavor`, including the 5-case compaction matrix (budget=0 disabled, budget=1 with newer-victim tiebreak, non-evictable bounds, priority-blocked, request priority overriding).

---

## Status

Beta-quality. Algorithm and snapshot integration are well-tested; the scheduler-level `issueTASCompaction` path uses the existing `workload.Evict` machinery (same channel as `issueMigration`) and requeues the pending workload for the next cycle.

Future work — likely upstream-able as KEP-2724 Story 4 ("rank-aware packing of pods"):

- Per-kind evictability defaults wired through the LWS / JobSet / Job integrations (so workloads don't need the annotation).
- Multi-ordered-level topologies (today only one ordered level per Topology is allowed).
- A `victimComparator` that delegates to `CandidatesOrdering` so compaction stays bit-consistent with cohort-wide preemption decisions.

---

## Upstream

Kept in sync with `kubernetes-sigs/kueue`. The `chain-adjacency` branch holds the patch; `main` tracks upstream.

```bash
git remote add upstream https://github.com/kubernetes-sigs/kueue.git
git fetch upstream
git rebase upstream/main chain-adjacency   # rebase the patch
```

For everything not specific to chain-adjacency (job integrations, fair sharing, MultiKueue, AdmissionChecks, etc.), see the [upstream Kueue documentation](https://kueue.sigs.k8s.io/).
