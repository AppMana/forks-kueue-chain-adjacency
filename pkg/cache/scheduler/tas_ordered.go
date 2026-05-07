/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Ordered topology allocator: a 1-D contiguous-range allocator on an ordered
// topology level. Used wherever rank-to-position adjacency matters for
// performance — examples include NVLink rings (Grace-Hopper C2C, AMD
// Infinity Fabric XGMI), Slingshot dragonfly inter-group links, NCCL ring
// collectives over fat-tree fabrics where consecutive ranks should land on
// physically-close nodes, and explicit daisy-chained fabrics (EtherCAT,
// PROFINET, PCIe expansion). Modelled as a compacting garbage collector
// where bound workloads play the role of live objects and free intervals
// are heap fragments. Allocations may be relocated within a caller-supplied
// budget to admit a pending request.

package scheduler

import (
	"cmp"
	"fmt"
	"math/bits"
	"sort"
)

// orderedAllocation is a workload bound to a contiguous run on an ordered
// topology level. The allocator works in pure positions (0..chainSize-1) and
// is unaware of Kueue-specific types; the snapshot integration translates.
type orderedAllocation struct {
	id        string
	start     int
	size      int
	priority  int32
	boundAt   int64
	evictable bool
}

func (a orderedAllocation) end() int { return a.start + a.size }

func (a orderedAllocation) overlapsRun(start, size int) bool {
	return a.start < start+size && start < a.end()
}

// orderedRequest is a pending placement request.
type orderedRequest struct {
	size     int
	priority int32
}

// orderedEviction tells the caller to relocate the named allocation to a new
// start. It is only emitted when newStart != the allocation's current start;
// no-op moves are filtered out before the plan is returned.
type orderedEviction struct {
	id       string
	newStart int
}

// orderedPlan is the result of a scheduling attempt. If pending is true, the
// request was not admitted and the caller should leave the workload pending.
type orderedPlan struct {
	pending       bool
	pendingReason string
	placement     int
	evictions     []orderedEviction
}

// victimComparator orders bound allocations by their cost-of-eviction.
// Returns negative if `a` is cheaper to evict than `b` (preferred for
// eviction); positive if `b` is cheaper; zero if equally expensive.
//
// The allocator uses this to compare entire eviction sets via lex-max:
// each plan's victim list is sorted descending (most expensive first), and
// two plans are compared element-by-element. A plan whose most-expensive
// victim is cheaper than the other plan's most-expensive victim wins;
// ties are broken on the next-most-expensive, etc. This minimises the
// maximum disruption first.
//
// When wired into Kueue, supply a comparator that calls
// pkg/scheduler/preemption/common/ordering.go:CandidatesOrdering, so chain
// compaction stays consistent with cohort-wide preemption decisions.
type victimComparator func(a, b orderedAllocation) int

// orderedAllocator solves contiguous-range allocation on an ordered level.
// It is invoked once per scheduling pass for a single pending request.
type orderedAllocator struct {
	chainSize int
	bound     []orderedAllocation
	// victimLess optionally overrides how victims are ranked. When nil the
	// allocator falls back to defaultVictimLess (priority asc, boundAt desc
	// to protect older workloads, id asc for determinism).
	victimLess victimComparator
}

const (
	// orderedBudgetNoCompact disables compaction; the allocator only tries
	// first-fit on the existing layout. This is the safe default.
	orderedBudgetNoCompact = 0

	// orderedBudgetOptimal lets the allocator consider relocating any number
	// of evictable bound allocations. For chain-sized inputs this is the
	// exhaustive optimum.
	orderedBudgetOptimal = -1

	// orderedSearchMaxEvictables caps the exhaustive subset enumeration. Above
	// this threshold the allocator refuses to attempt compaction, since
	// 2^N grows quickly. In practice chain-sized inputs are well below this.
	orderedSearchMaxEvictables = 24
)

// schedule returns a placement plan for req under the given budget.
//
//	budget == 0   : never compact (first-fit only).
//	budget == -1  : exhaustive optimal search (subject to a safety cap).
//	budget >  0   : compact at most this many bound allocations.
//
// Pure: no I/O, no Kueue dependency, deterministic.
func (a *orderedAllocator) schedule(req orderedRequest, budget int) orderedPlan {
	if req.size <= 0 {
		return orderedPlan{
			pending:       true,
			pendingReason: fmt.Sprintf("request size must be positive, got %d", req.size),
		}
	}
	if req.size > a.chainSize {
		return orderedPlan{
			pending: true,
			pendingReason: fmt.Sprintf("request size %d exceeds chain size %d",
				req.size, a.chainSize),
		}
	}
	// First-fit on existing layout (zero-disruption path).
	if start, ok := firstFitLeftmost(a.bound, a.chainSize, req.size); ok {
		return orderedPlan{placement: start}
	}
	if budget == orderedBudgetNoCompact {
		return orderedPlan{
			pending:       true,
			pendingReason: "no contiguous free run; compaction disabled (budget=0)",
		}
	}
	return a.solveWithCompaction(req, budget)
}

// firstFitLeftmost finds the smallest start such that [start, start+size) is
// fully free given the supplied bound set. Returns (start, true) on success.
// O(chainSize) — trivially fast for any realistic chain.
func firstFitLeftmost(bound []orderedAllocation, chainSize, size int) (int, bool) {
	if size <= 0 || size > chainSize {
		return 0, false
	}
	occupied := make([]bool, chainSize)
	for _, b := range bound {
		lo, hi := b.start, b.end()
		if lo < 0 {
			lo = 0
		}
		if hi > chainSize {
			hi = chainSize
		}
		for i := lo; i < hi; i++ {
			occupied[i] = true
		}
	}
	runLen := 0
	for i := 0; i < chainSize; i++ {
		if occupied[i] {
			runLen = 0
			continue
		}
		runLen++
		if runLen >= size {
			return i - size + 1, true
		}
	}
	return 0, false
}

// solveWithCompaction performs the exhaustive search, capped by budget.
// Iterates over all subsets of evictable allocations sized ≤ budget;
// for each subset, attempts to place the request among the staying
// allocations, then relocates the evicted set into the resulting free space.
// Picks the lowest-cost feasible plan (see comparePlan).
func (a *orderedAllocator) solveWithCompaction(req orderedRequest, budget int) orderedPlan {
	// Partition bounds into evictables (subject to req's priority) and pinned
	// (non-evictable OR strictly higher-priority than req).
	var evictables, pinned []orderedAllocation
	for _, b := range a.bound {
		if b.evictable && b.priority <= req.priority {
			evictables = append(evictables, b)
		} else {
			pinned = append(pinned, b)
		}
	}
	// Deterministic order so subset enumeration is reproducible.
	sort.Slice(evictables, func(i, j int) bool {
		if evictables[i].priority != evictables[j].priority {
			return evictables[i].priority < evictables[j].priority
		}
		if evictables[i].boundAt != evictables[j].boundAt {
			return evictables[i].boundAt < evictables[j].boundAt
		}
		return evictables[i].id < evictables[j].id
	})

	n := len(evictables)
	if n > orderedSearchMaxEvictables {
		return orderedPlan{
			pending: true,
			pendingReason: fmt.Sprintf(
				"too many evictable allocations (%d) for exhaustive solver; cap=%d",
				n, orderedSearchMaxEvictables),
		}
	}
	maxSubsetSize := n
	if budget != orderedBudgetOptimal && budget < n {
		maxSubsetSize = budget
	}
	oldStart := make(map[string]int, len(a.bound))
	for _, b := range a.bound {
		oldStart[b.id] = b.start
	}
	byID := make(map[string]orderedAllocation, len(a.bound))
	for _, b := range a.bound {
		byID[b.id] = b
	}

	var best *orderedPlan
	var bestSortedVictims []orderedAllocation
	var bestRelocDist int

	for mask := 0; mask < 1<<n; mask++ {
		if bits.OnesCount(uint(mask)) > maxSubsetSize {
			continue
		}
		var evictSet, keepEvictables []orderedAllocation
		for i := 0; i < n; i++ {
			if mask&(1<<i) != 0 {
				evictSet = append(evictSet, evictables[i])
			} else {
				keepEvictables = append(keepEvictables, evictables[i])
			}
		}
		staying := make([]orderedAllocation, 0, len(pinned)+len(keepEvictables))
		staying = append(staying, pinned...)
		staying = append(staying, keepEvictables...)

		plan, ok := tryPlanWithEvictions(req, evictSet, staying, oldStart, a.chainSize)
		if !ok {
			continue
		}
		// Effective eviction count must still respect the budget after no-op
		// moves are filtered out (a relocation that resolves to the same start
		// is not a real eviction).
		if budget != orderedBudgetOptimal && len(plan.evictions) > budget {
			continue
		}
		sortedVictims := a.sortedVictimsDesc(plan.evictions, byID)
		relocDist := totalRelocationDistance(plan, oldStart, byID)
		if best == nil ||
			a.comparePrecomputed(plan, sortedVictims, relocDist, *best, bestSortedVictims, bestRelocDist) < 0 {
			cp := plan
			best = &cp
			bestSortedVictims = sortedVictims
			bestRelocDist = relocDist
		}
	}
	if best == nil {
		reason := "no feasible plan exists for this request"
		if budget != orderedBudgetOptimal {
			reason = fmt.Sprintf("no feasible plan within budget %d", budget)
		}
		return orderedPlan{pending: true, pendingReason: reason}
	}
	return *best
}

// tryPlanWithEvictions attempts to construct a layout where:
//   - staying allocations keep their current positions,
//   - the request gets a contiguous run somewhere not overlapping staying,
//   - the evictSet is re-placed into the remaining free space.
//
// Returns (plan, true) on success. The plan's evictions list omits no-op moves
// (where an evicted allocation lands on its original start).
func tryPlanWithEvictions(
	req orderedRequest,
	evictSet, staying []orderedAllocation,
	oldStart map[string]int,
	chainSize int,
) (orderedPlan, bool) {
	start, ok := firstFitLeftmost(staying, chainSize, req.size)
	if !ok {
		return orderedPlan{}, false
	}
	occupied := make([]orderedAllocation, 0, len(staying)+1)
	occupied = append(occupied, staying...)
	occupied = append(occupied, orderedAllocation{
		id:    "__request__",
		start: start,
		size:  req.size,
	})
	free := computeFreeIntervals(occupied, chainSize)
	relocs, ok := placeFirstFitDecreasing(evictSet, free)
	if !ok {
		return orderedPlan{}, false
	}
	evictions := make([]orderedEviction, 0, len(relocs))
	for _, r := range relocs {
		if r.newStart != oldStart[r.id] {
			evictions = append(evictions, orderedEviction{
				id:       r.id,
				newStart: r.newStart,
			})
		}
	}
	sort.Slice(evictions, func(i, j int) bool {
		if evictions[i].newStart != evictions[j].newStart {
			return evictions[i].newStart < evictions[j].newStart
		}
		return evictions[i].id < evictions[j].id
	})
	return orderedPlan{placement: start, evictions: evictions}, true
}

// interval is a half-open chain-index range [start, end).
type interval struct{ start, end int }

func (iv interval) length() int { return iv.end - iv.start }

// computeFreeIntervals returns the maximal contiguous free runs in
// [0, chainSize) given a set of bound allocations. Bounds may overlap with the
// chain bounds; the function clamps to the chain. Result is in increasing
// start order, no zero-length runs.
func computeFreeIntervals(bound []orderedAllocation, chainSize int) []interval {
	sorted := make([]orderedAllocation, len(bound))
	copy(sorted, bound)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].start < sorted[j].start
	})
	result := make([]interval, 0, len(sorted)+1)
	cursor := 0
	for _, b := range sorted {
		bs, be := b.start, b.end()
		if bs < 0 {
			bs = 0
		}
		if be > chainSize {
			be = chainSize
		}
		if bs > cursor {
			result = append(result, interval{start: cursor, end: bs})
		}
		if be > cursor {
			cursor = be
		}
		if cursor >= chainSize {
			break
		}
	}
	if cursor < chainSize {
		result = append(result, interval{start: cursor, end: chainSize})
	}
	return result
}

// relocation is the new start the allocator picked for an evicted allocation.
type relocation struct {
	id       string
	newStart int
}

// placeFirstFitDecreasing tries to place each allocation in one of the free
// intervals using first-fit-decreasing (largest first). Returns the chosen
// starts; ok=false if any allocation doesn't fit.
//
// First-fit-decreasing is well-studied as a 2-competitive heuristic for 1-D
// bin-packing, which is sufficient for our chain-sized inputs. The outer
// solver enumerates subsets exhaustively, so even when this inner heuristic
// fails on a particular subset, another subset choice may succeed.
func placeFirstFitDecreasing(allocs []orderedAllocation, free []interval) ([]relocation, bool) {
	if len(allocs) == 0 {
		return nil, true
	}
	sorted := make([]orderedAllocation, len(allocs))
	copy(sorted, allocs)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].size != sorted[j].size {
			return sorted[i].size > sorted[j].size
		}
		if sorted[i].priority != sorted[j].priority {
			return sorted[i].priority > sorted[j].priority
		}
		return sorted[i].id < sorted[j].id
	})
	remaining := make([]interval, len(free))
	copy(remaining, free)
	result := make([]relocation, 0, len(sorted))
	for _, a := range sorted {
		placed := false
		for i, iv := range remaining {
			if iv.length() >= a.size {
				result = append(result, relocation{id: a.id, newStart: iv.start})
				remaining[i] = interval{start: iv.start + a.size, end: iv.end}
				placed = true
				break
			}
		}
		if !placed {
			return nil, false
		}
	}
	return result, true
}

// defaultVictimLess ranks bound allocations by cost-of-eviction when no
// integration-supplied comparator is present. Lower priority is cheaper to
// evict; among equal priorities, newer (higher boundAt) is cheaper, so older
// workloads — which have invested more in setup (warm models, established
// NCCL rings) — are protected. The id is the final deterministic tiebreak.
func defaultVictimLess(a, b orderedAllocation) int {
	if c := cmp.Compare(a.priority, b.priority); c != 0 {
		return c
	}
	if c := cmp.Compare(b.boundAt, a.boundAt); c != 0 {
		return c
	}
	return cmp.Compare(a.id, b.id)
}

// sortedVictimsDesc looks up the orderedAllocation for each evicted id and
// returns the list sorted *descending* by victimLess (most expensive first).
// This is the canonical form for lex-max comparison of two plans' victim
// sets.
func (a *orderedAllocator) sortedVictimsDesc(
	evictions []orderedEviction,
	byID map[string]orderedAllocation,
) []orderedAllocation {
	if len(evictions) == 0 {
		return nil
	}
	less := a.victimLess
	if less == nil {
		less = defaultVictimLess
	}
	result := make([]orderedAllocation, 0, len(evictions))
	for _, e := range evictions {
		if alloc, ok := byID[e.id]; ok {
			result = append(result, alloc)
		}
	}
	sort.Slice(result, func(i, j int) bool {
		return less(result[i], result[j]) > 0
	})
	return result
}

// totalRelocationDistance is the sum of |newStart - oldStart| over all
// real (non-no-op) evictions in the plan. Used as a cosmetic discriminator
// among plans with equal victim quality: prefer plans that physically move
// fewer cells, which tends to reduce NCCL re-init churn for the moved
// workloads.
func totalRelocationDistance(
	p orderedPlan,
	oldStart map[string]int,
	byID map[string]orderedAllocation,
) int {
	if len(p.evictions) == 0 {
		return 0
	}
	total := 0
	for _, e := range p.evictions {
		old, ok := oldStart[e.id]
		if !ok {
			continue
		}
		_ = byID // reserved for future per-victim weighting (e.g. resource cost)
		d := e.newStart - old
		if d < 0 {
			d = -d
		}
		total += d
	}
	return total
}

// comparePrecomputed is the plan-level comparator. Negative if `a` is
// preferred over `b`. Compares in this order:
//
//  1. Eviction count (fewer is better — disruption is the primary cost).
//  2. Lex-max victim ordering. Both plans' victims are pre-sorted descending
//     by victimLess; we compare element-by-element. The plan whose
//     most-expensive victim is cheaper than the other plan's most-expensive
//     wins; ties on the next-most-expensive, etc. Minimises maximum
//     disruption first.
//  3. Total relocation distance — among equal-quality victim sets, prefer
//     plans that move fewer cells.
//  4. Leftmost placement — deterministic final tiebreak.
//
// The pre-sorted victim lists are passed in to avoid re-sorting per
// comparison in the hot loop.
func (a *orderedAllocator) comparePrecomputed(
	planA orderedPlan, victimsA []orderedAllocation, distA int,
	planB orderedPlan, victimsB []orderedAllocation, distB int,
) int {
	if c := cmp.Compare(len(planA.evictions), len(planB.evictions)); c != 0 {
		return c
	}
	less := a.victimLess
	if less == nil {
		less = defaultVictimLess
	}
	for i := 0; i < len(victimsA); i++ {
		if c := less(victimsA[i], victimsB[i]); c != 0 {
			return c
		}
	}
	if c := cmp.Compare(distA, distB); c != 0 {
		return c
	}
	return cmp.Compare(planA.placement, planB.placement)
}
