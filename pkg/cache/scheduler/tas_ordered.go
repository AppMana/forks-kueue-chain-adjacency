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
	id    string
	start int
	// size is the number of slices the workload holds — the length of the
	// contiguous run it needs when (re-)placed. It is NOT derived from the
	// span of `positions`: a fragmented workload occupying {0,2} has
	// size 2, not 3.
	size int
	// positions optionally lists the exact chain positions the allocation
	// currently occupies, sorted ascending. nil means the contiguous run
	// [start, start+size). A non-contiguous positions list models a
	// workload whose pods drifted apart (e.g. LWS pod recreation during
	// rolling node reboots); compaction re-places it contiguously.
	positions []int
	priority  int32
	boundAt   int64
	evictable bool
}

func (a orderedAllocation) end() int { return a.start + a.size }

// occupied returns the chain positions the allocation currently holds.
func (a orderedAllocation) occupied() []int {
	if a.positions != nil {
		return a.positions
	}
	out := make([]int, 0, a.size)
	for i := a.start; i < a.start+a.size; i++ {
		out = append(out, i)
	}
	return out
}

// isContiguousAt reports whether the allocation already occupies exactly
// the contiguous run [start, start+size). Used to filter no-op relocations:
// re-placing a fragmented allocation is always a real eviction, even when
// the new run begins at the old start position.
func (a orderedAllocation) isContiguousAt(start int) bool {
	if a.positions == nil {
		return a.start == start
	}
	if len(a.positions) != a.size {
		return false
	}
	for i, p := range a.positions {
		if p != start+i {
			return false
		}
	}
	return true
}

func (a orderedAllocation) overlapsRun(start, size int) bool {
	for _, p := range a.occupied() {
		if p >= start && p < start+size {
			return true
		}
	}
	return false
}

// orderedRequest is a pending placement request.
type orderedRequest struct {
	size     int
	priority int32
	// preferredPositions lists chain positions the requesting workload
	// previously occupied (its last-known assignment). Placement prefers
	// runs overlapping these positions — weights and compile caches are
	// node-local, so returning to previous nodes avoids a cold reload.
	// A scoring preference only, never a hard constraint. nil disables it.
	preferredPositions []int
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
	// Zero-disruption path: place into the existing layout. Among the
	// candidate starts, prefer the one covering the most preferred (warm)
	// positions; ties resolve leftmost, which reduces to plain first-fit
	// when no preference is supplied.
	free := computeFreeIntervals(a.bound, a.chainSize)
	if starts := requestStartCandidates(free, req.size, req.preferredPositions); len(starts) > 0 {
		best := starts[0]
		bestOverlap := overlapWithRun(req.preferredPositions, best, req.size)
		for _, s := range starts[1:] {
			if ov := overlapWithRun(req.preferredPositions, s, req.size); ov > bestOverlap {
				best, bestOverlap = s, ov
			}
		}
		return orderedPlan{placement: best}
	}
	if budget == orderedBudgetNoCompact {
		return orderedPlan{
			pending:       true,
			pendingReason: "no contiguous free run; compaction disabled (budget=0)",
		}
	}
	return a.solveWithCompaction(req, budget)
}

// occupiedMask marks every chain position held by the given allocations,
// clamped to [0, chainSize). Fragmented allocations mark exactly their
// occupied positions — position gaps inside a fragmented workload stay free.
func occupiedMask(bound []orderedAllocation, chainSize int) []bool {
	occupied := make([]bool, chainSize)
	for _, b := range bound {
		for _, p := range b.occupied() {
			if p >= 0 && p < chainSize {
				occupied[p] = true
			}
		}
	}
	return occupied
}

// firstFitLeftmost finds the smallest start such that [start, start+size) is
// fully free given the supplied bound set. Returns (start, true) on success.
// O(chainSize) — trivially fast for any realistic chain.
func firstFitLeftmost(bound []orderedAllocation, chainSize, size int) (int, bool) {
	if size <= 0 || size > chainSize {
		return 0, false
	}
	occupied := occupiedMask(bound, chainSize)
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

// requestStartCandidates returns deterministic candidate starts for a
// size-cell contiguous run in the given free intervals: each interval's
// leftmost and rightmost feasible start (the only Pareto-optimal choices
// for keeping the remaining free space contiguous), plus a warm anchor at
// the request's first preferred position when a run fits there. Sorted
// ascending.
func requestStartCandidates(free []interval, size int, preferred []int) []int {
	var out []int
	seen := make(map[int]bool, 3*len(free))
	add := func(s int) {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	for _, iv := range free {
		if iv.length() < size {
			continue
		}
		add(iv.start)
		add(iv.end - size)
		if len(preferred) > 0 {
			if anchor := preferred[0]; anchor >= iv.start && anchor+size <= iv.end {
				add(anchor)
			}
		}
	}
	sort.Ints(out)
	return out
}

// overlapWithRun counts how many of the given positions fall inside the
// run [start, start+size) — the cache-warmth score of a placement.
func overlapWithRun(positions []int, start, size int) int {
	n := 0
	for _, p := range positions {
		if p >= start && p < start+size {
			n++
		}
	}
	return n
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
	var bestWarmth int

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

		for _, plan := range tryPlansWithEvictions(req, evictSet, staying, byID, a.chainSize) {
			// Effective eviction count must still respect the budget after
			// no-op moves are filtered out (a relocation that leaves an
			// allocation contiguous at its current start is not a real
			// eviction).
			if budget != orderedBudgetOptimal && len(plan.evictions) > budget {
				continue
			}
			sortedVictims := a.sortedVictimsDesc(plan.evictions, byID)
			relocDist := totalRelocationDistance(plan, oldStart, byID)
			warmth := planWarmth(plan, req, byID)
			if best == nil ||
				a.comparePrecomputed(plan, sortedVictims, relocDist, warmth, *best, bestSortedVictims, bestRelocDist, bestWarmth) < 0 {
				cp := plan
				best = &cp
				bestSortedVictims = sortedVictims
				bestRelocDist = relocDist
				bestWarmth = warmth
			}
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

// tryPlansWithEvictions constructs every candidate layout for one eviction
// subset:
//   - staying allocations keep their current positions,
//   - the request gets a contiguous run at one of the deterministic
//     candidate starts (interval corners plus the warm anchor),
//   - the evictSet is re-placed into the remaining free space, preferring
//     runs overlapping each victim's previous positions.
//
// One plan is returned per feasible request start; the caller scores them.
// Each plan's evictions list omits no-op moves — but re-placing a
// FRAGMENTED allocation is always a real eviction, even when its new run
// begins at its old start position ({0,2} → {0,1} moves the pod at 2).
func tryPlansWithEvictions(
	req orderedRequest,
	evictSet, staying []orderedAllocation,
	byID map[string]orderedAllocation,
	chainSize int,
) []orderedPlan {
	freeForRequest := computeFreeIntervals(staying, chainSize)
	starts := requestStartCandidates(freeForRequest, req.size, req.preferredPositions)
	var plans []orderedPlan
	for _, start := range starts {
		occupied := make([]orderedAllocation, 0, len(staying)+1)
		occupied = append(occupied, staying...)
		occupied = append(occupied, orderedAllocation{
			id:    "__request__",
			start: start,
			size:  req.size,
		})
		free := computeFreeIntervals(occupied, chainSize)
		relocs, ok := placeEvictedWarm(evictSet, free)
		if !ok {
			continue
		}
		evictions := make([]orderedEviction, 0, len(relocs))
		for _, r := range relocs {
			if !byID[r.id].isContiguousAt(r.newStart) {
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
		plans = append(plans, orderedPlan{placement: start, evictions: evictions})
	}
	return plans
}

// planWarmth scores a plan's cache warmth: the number of preferred (warm)
// positions the request's placement covers, plus — for every relocated
// victim — the number of the victim's current positions its new run keeps.
// Higher is better: every warm cell is a node that skips a weight reload.
func planWarmth(p orderedPlan, req orderedRequest, byID map[string]orderedAllocation) int {
	total := overlapWithRun(req.preferredPositions, p.placement, req.size)
	for _, e := range p.evictions {
		if b, ok := byID[e.id]; ok {
			total += overlapWithRun(b.occupied(), e.newStart, b.size)
		}
	}
	return total
}

// interval is a half-open chain-index range [start, end).
type interval struct{ start, end int }

func (iv interval) length() int { return iv.end - iv.start }

// computeFreeIntervals returns the maximal contiguous free runs in
// [0, chainSize) given a set of bound allocations. Bounds may overlap with
// the chain bounds; the function clamps to the chain. Fragmented
// allocations free the positions inside their span they do not actually
// occupy. Result is in increasing start order, no zero-length runs.
func computeFreeIntervals(bound []orderedAllocation, chainSize int) []interval {
	occupied := occupiedMask(bound, chainSize)
	var result []interval
	runStart := -1
	for i := 0; i <= chainSize; i++ {
		free := i < chainSize && !occupied[i]
		if free && runStart < 0 {
			runStart = i
		}
		if !free && runStart >= 0 {
			result = append(result, interval{start: runStart, end: i})
			runStart = -1
		}
	}
	return result
}

// relocation is the new start the allocator picked for an evicted allocation.
type relocation struct {
	id       string
	newStart int
}

// placeEvictedWarm places each evicted allocation into the free intervals
// in first-fit-decreasing order (largest first — the classic 2-competitive
// 1-D bin-packing heuristic), preferring for each allocation the placement
// that keeps the most of its previous positions (cache warmth), tiebroken
// leftmost. When the warm-greedy pass fails to fit every allocation it
// falls back to plain leftmost first-fit-decreasing, so warmth never costs
// feasibility. The outer solver enumerates eviction subsets exhaustively,
// so even when both passes fail on a particular subset, another subset
// choice may succeed.
func placeEvictedWarm(allocs []orderedAllocation, free []interval) ([]relocation, bool) {
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
	if result, ok := placeWithSelector(sorted, free, warmPlacement); ok {
		return result, true
	}
	return placeWithSelector(sorted, free, leftmostPlacement)
}

// placementSelector picks (interval index, offset) for one allocation from
// the remaining free intervals; ok=false if it fits nowhere.
type placementSelector func(a orderedAllocation, remaining []interval) (ivIdx, offset int, ok bool)

// leftmostPlacement is the plain first-fit choice: the start of the first
// interval large enough.
func leftmostPlacement(a orderedAllocation, remaining []interval) (int, int, bool) {
	for i, iv := range remaining {
		if iv.length() >= a.size {
			return i, iv.start, true
		}
	}
	return 0, 0, false
}

// warmPlacement picks, among each fitting interval's corner offsets and a
// warm anchor at the allocation's first previous position, the placement
// covering the most of the allocation's previous positions; ties resolve
// to the smallest offset. With zero overlap everywhere this reduces to
// leftmostPlacement (intervals are in ascending order and iv.start of the
// first fitting interval is the smallest candidate offset).
func warmPlacement(a orderedAllocation, remaining []interval) (int, int, bool) {
	prev := a.occupied()
	bestIdx, bestOffset, bestOverlap, found := 0, 0, -1, false
	consider := func(idx, offset int) {
		ov := overlapWithRun(prev, offset, a.size)
		if !found || ov > bestOverlap {
			bestIdx, bestOffset, bestOverlap, found = idx, offset, ov, true
		}
	}
	for i, iv := range remaining {
		if iv.length() < a.size {
			continue
		}
		consider(i, iv.start)
		consider(i, iv.end-a.size)
		if len(prev) > 0 {
			if anchor := prev[0]; anchor >= iv.start && anchor+a.size <= iv.end {
				consider(i, anchor)
			}
		}
	}
	return bestIdx, bestOffset, found
}

// placeWithSelector runs the placement loop with the given selector,
// splitting intervals around mid-interval placements.
func placeWithSelector(sorted []orderedAllocation, free []interval, pick placementSelector) ([]relocation, bool) {
	remaining := make([]interval, len(free))
	copy(remaining, free)
	result := make([]relocation, 0, len(sorted))
	for _, a := range sorted {
		idx, offset, ok := pick(a, remaining)
		if !ok {
			return nil, false
		}
		result = append(result, relocation{id: a.id, newStart: offset})
		iv := remaining[idx]
		next := make([]interval, 0, len(remaining)+1)
		next = append(next, remaining[:idx]...)
		if offset > iv.start {
			next = append(next, interval{start: iv.start, end: offset})
		}
		if offset+a.size < iv.end {
			next = append(next, interval{start: offset + a.size, end: iv.end})
		}
		next = append(next, remaining[idx+1:]...)
		remaining = next
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
//  3. Cache warmth — among equal-disruption plans, prefer the one keeping
//     more workload cells on their previous (warm) positions. Weights and
//     compile caches are node-local; every warm cell skips a reload.
//  4. Total relocation distance — among equal-warmth victim sets, prefer
//     plans that move fewer cells.
//  5. Leftmost placement — deterministic final tiebreak.
//
// The pre-sorted victim lists, distances and warmth scores are passed in to
// avoid recomputing per comparison in the hot loop.
func (a *orderedAllocator) comparePrecomputed(
	planA orderedPlan, victimsA []orderedAllocation, distA, warmA int,
	planB orderedPlan, victimsB []orderedAllocation, distB, warmB int,
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
	if c := cmp.Compare(warmB, warmA); c != 0 { // higher warmth preferred
		return c
	}
	if c := cmp.Compare(distA, distB); c != 0 {
		return c
	}
	return cmp.Compare(planA.placement, planB.placement)
}
