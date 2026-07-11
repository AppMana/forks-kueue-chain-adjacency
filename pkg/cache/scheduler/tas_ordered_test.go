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

// Tests for the ordered (compacting-GC-style) topology allocator.
// The scenarios mirror classic memory-allocator test suites:
// fragmentation patterns (checkerboard, sequential), coalescing,
// boundary conditions, worst-case fragmentation, optimal-vs-heuristic
// parity, invariant checking on every plan.

package scheduler

import (
	"cmp"
	"fmt"
	"math/rand"
	"slices"
	"sort"
	"testing"
)

// alloc is a test-helper builder for orderedAllocation.
func alloc(id string, start, size int) orderedAllocation {
	return orderedAllocation{id: id, start: start, size: size, evictable: true, boundAt: 1}
}

func allocPrio(id string, start, size int, priority int32) orderedAllocation {
	return orderedAllocation{id: id, start: start, size: size, priority: priority, evictable: true, boundAt: 1}
}

func allocPinned(id string, start, size int) orderedAllocation {
	return orderedAllocation{id: id, start: start, size: size, evictable: false, boundAt: 1}
}

// schedulerTestCase drives schedule() with a starting layout and a request.
type schedulerTestCase struct {
	name      string
	chainSize int
	bound     []orderedAllocation
	req       orderedRequest
	budget    int

	// Expectations:
	wantPending      bool
	wantPlacement    int
	wantEvictionIDs  []string // sorted; empty slice means no evictions
	wantEvictNewSets map[string]int
}

func runSchedulerTestCase(t *testing.T, c schedulerTestCase) {
	t.Helper()
	a := &orderedAllocator{chainSize: c.chainSize, bound: c.bound}
	plan := a.schedule(c.req, c.budget)

	if plan.pending != c.wantPending {
		t.Fatalf("pending = %v (reason=%q), want %v",
			plan.pending, plan.pendingReason, c.wantPending)
	}
	if plan.pending {
		return
	}
	if plan.placement != c.wantPlacement {
		t.Errorf("placement = %d, want %d", plan.placement, c.wantPlacement)
	}
	gotIDs := make([]string, 0, len(plan.evictions))
	for _, e := range plan.evictions {
		gotIDs = append(gotIDs, e.id)
	}
	sort.Strings(gotIDs)
	wantIDs := slices.Clone(c.wantEvictionIDs)
	sort.Strings(wantIDs)
	if !slices.Equal(gotIDs, wantIDs) {
		t.Errorf("evicted ids = %v, want %v", gotIDs, wantIDs)
	}
	if c.wantEvictNewSets != nil {
		for _, e := range plan.evictions {
			if want, ok := c.wantEvictNewSets[e.id]; ok && e.newStart != want {
				t.Errorf("eviction %s newStart = %d, want %d", e.id, e.newStart, want)
			}
		}
	}
	checkPlanInvariants(t, c.chainSize, c.bound, c.req, plan)
}

// checkPlanInvariants verifies a plan is internally consistent. Run after
// every successful schedule call. Catches bugs the explicit assertions miss.
func checkPlanInvariants(t *testing.T, chainSize int, bound []orderedAllocation, req orderedRequest, plan orderedPlan) {
	t.Helper()
	if plan.pending {
		return
	}
	// 1. Request fits in chain.
	if plan.placement < 0 || plan.placement+req.size > chainSize {
		t.Errorf("invariant: request out of bounds: placement=%d size=%d chain=%d",
			plan.placement, req.size, chainSize)
	}
	// 2. Evictions only reference real allocations, all evictable, all priority ≤ req.
	byID := make(map[string]orderedAllocation, len(bound))
	for _, b := range bound {
		byID[b.id] = b
	}
	evictedTo := make(map[string]int, len(plan.evictions))
	for _, e := range plan.evictions {
		b, ok := byID[e.id]
		if !ok {
			t.Errorf("invariant: eviction references unknown allocation %q", e.id)
			continue
		}
		if !b.evictable {
			t.Errorf("invariant: eviction moves non-evictable allocation %q", e.id)
		}
		if b.priority > req.priority {
			t.Errorf("invariant: eviction moves higher-priority allocation %q (prio=%d > req.prio=%d)",
				e.id, b.priority, req.priority)
		}
		if e.newStart < 0 || e.newStart+b.size > chainSize {
			t.Errorf("invariant: eviction %q out of bounds: newStart=%d size=%d", e.id, e.newStart, b.size)
		}
		if b.isContiguousAt(e.newStart) {
			t.Errorf("invariant: eviction %q is a no-op (already contiguous at %d)", e.id, e.newStart)
		}
		evictedTo[e.id] = e.newStart
	}
	// 3. No overlaps in final layout (request + all bounds at their final
	// positions). Evicted allocations occupy their new contiguous run;
	// untouched allocations occupy their current (possibly fragmented)
	// positions.
	occupancy := make(map[int]string, chainSize)
	for i := plan.placement; i < plan.placement+req.size; i++ {
		occupancy[i] = "__request__"
	}
	for _, b := range bound {
		var cells []int
		if s, ok := evictedTo[b.id]; ok {
			for i := s; i < s+b.size; i++ {
				cells = append(cells, i)
			}
		} else {
			cells = b.occupied()
		}
		for _, i := range cells {
			if existing, found := occupancy[i]; found {
				t.Errorf("invariant: overlap at chain-index %d between %q and %q", i, existing, b.id)
			}
			occupancy[i] = b.id
		}
	}
}

// ---------------------------------------------------------------------------
// Section 1: Allocation correctness — empty chain, sequential, full chain.
// ---------------------------------------------------------------------------

func TestOrderedAllocator_EmptyChain(t *testing.T) {
	runSchedulerTestCase(t, schedulerTestCase{
		name:          "empty chain admits at 0",
		chainSize:     12,
		req:           orderedRequest{size: 8},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 0,
	})
}

func TestOrderedAllocator_TwoCoexist(t *testing.T) {
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "after A=8@0..7 place B=4 at 8..11",
		chainSize: 12,
		bound:     []orderedAllocation{alloc("A", 0, 8)},
		req:       orderedRequest{size: 4},
		budget:    orderedBudgetNoCompact,

		wantPlacement: 8,
	})
}

func TestOrderedAllocator_FullChainNoFit(t *testing.T) {
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "fully occupied chain rejects size=1 with budget=0",
		chainSize: 12,
		bound: []orderedAllocation{
			alloc("A", 0, 8),
			alloc("B", 8, 4),
		},
		req:         orderedRequest{size: 1},
		budget:      orderedBudgetNoCompact,
		wantPending: true,
	})
}

// ---------------------------------------------------------------------------
// Section 2: First-fit on fragmented layouts (no compaction).
// ---------------------------------------------------------------------------

func TestOrderedAllocator_FirstFitChoosesLeftmost(t *testing.T) {
	// idle 0..3, B=2@4..5, idle 6..11; req=4 → fits at 0..3 (leftmost).
	runSchedulerTestCase(t, schedulerTestCase{
		name:          "first-fit picks leftmost run",
		chainSize:     12,
		bound:         []orderedAllocation{alloc("B", 4, 2)},
		req:           orderedRequest{size: 4},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 0,
	})
}

func TestOrderedAllocator_FirstFitSkipsTooSmall(t *testing.T) {
	// idle 0..3, B=2@4..5, idle 6..11; req=5 → only 6-cell gap fits.
	runSchedulerTestCase(t, schedulerTestCase{
		name:          "first-fit skips too-small run",
		chainSize:     12,
		bound:         []orderedAllocation{alloc("B", 4, 2)},
		req:           orderedRequest{size: 5},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 6,
	})
}

func TestOrderedAllocator_NoFit_BudgetZero(t *testing.T) {
	// idle 0..3, B=2@4..5, idle 6..11; req=8 — no contiguous 8-cell run.
	runSchedulerTestCase(t, schedulerTestCase{
		name:        "no contiguous fit, budget=0 → pending",
		chainSize:   12,
		bound:       []orderedAllocation{alloc("B", 4, 2)},
		req:         orderedRequest{size: 8},
		budget:      orderedBudgetNoCompact,
		wantPending: true,
	})
}

// ---------------------------------------------------------------------------
// Section 3: Compaction scenarios (the user's plan scenarios + extras).
// ---------------------------------------------------------------------------

func TestOrderedAllocator_CompactSingleEvict(t *testing.T) {
	// The user's plan scenario #5: idle 0..3, B=2@4..5, idle 6..11;
	// req=8 with budget=1.  The minimum-eviction plan evicts B; the
	// allocator picks the leftmost placement that survives the eviction,
	// so the request lands at 0..7 and B is re-placed in the trailing gap
	// at 8..9.  Either layout (req@0..7 with B→8 or req@2..9 with B→0)
	// has eviction count 1; the leftmost-placement tiebreaker picks
	// req@0..7.
	runSchedulerTestCase(t, schedulerTestCase{
		name:             "compact one neighbor to free contiguous run",
		chainSize:        12,
		bound:            []orderedAllocation{alloc("B", 4, 2)},
		req:              orderedRequest{size: 8},
		budget:           1,
		wantPlacement:    0,
		wantEvictionIDs:  []string{"B"},
		wantEvictNewSets: map[string]int{"B": 8},
	})
}

func TestOrderedAllocator_ScaleUpDoesNotEvict(t *testing.T) {
	// Plan scenario #3: A=8@0..7, B=4@8..11; A scale to 12 (i.e. new request
	// from A for size=12).  Even with budget allowing it, we'd need to evict
	// B AND remove A — but the request comes from "A growing", which we model
	// as: A still bound at 0..7, request size=12.  No fit at any start since
	// any start contains either A or B.  Budget=0 → Pending.  This verifies
	// the user's "scale-up never evicts neighbours" requirement implicitly
	// by way of budget=0 default.
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "full chain, scale-up request stays pending at budget=0",
		chainSize: 12,
		bound: []orderedAllocation{
			alloc("A", 0, 8),
			alloc("B", 8, 4),
		},
		req:         orderedRequest{size: 12},
		budget:      orderedBudgetNoCompact,
		wantPending: true,
	})
}

func TestOrderedAllocator_PendingRequestLargerThanChain(t *testing.T) {
	runSchedulerTestCase(t, schedulerTestCase{
		name:        "request larger than chain → pending",
		chainSize:   12,
		req:         orderedRequest{size: 13},
		budget:      orderedBudgetOptimal,
		wantPending: true,
	})
}

func TestOrderedAllocator_RequestZeroOrNegative(t *testing.T) {
	for _, s := range []int{0, -1, -42} {
		c := schedulerTestCase{
			name:        fmt.Sprintf("request size %d → pending", s),
			chainSize:   12,
			req:         orderedRequest{size: s},
			budget:      orderedBudgetOptimal,
			wantPending: true,
		}
		runSchedulerTestCase(t, c)
	}
}

// ---------------------------------------------------------------------------
// Section 4: Pinning.
// ---------------------------------------------------------------------------

func TestOrderedAllocator_PinnedBlocksEviction(t *testing.T) {
	// A pinned at 0..3, B evictable at 4..5 — A overlaps any req start ≤ 3.
	// req size=4 with budget=10 → first-fit at 6..9 (no eviction needed, B
	// stays where it is, the 6..11 gap is size=6 so 6..9 fits the req).
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "first-fit succeeds without disturbing anything",
		chainSize: 12,
		bound: []orderedAllocation{
			allocPinned("A", 0, 4),
			alloc("B", 4, 2),
		},
		req:           orderedRequest{size: 4},
		budget:        10,
		wantPlacement: 6,
	})
}

func TestOrderedAllocator_PinnedSplitsChain(t *testing.T) {
	// A pinned at 2..3 splits chain. B evictable at 8..9. Idle elsewhere.
	// req size=8: only fitting placement is 4..11 (avoids A). Must evict B.
	// Re-place B in available [0..1].  Plan: req @ 4..11, B → 0..1.
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "pinned mid-chain forces compaction around it",
		chainSize: 12,
		bound: []orderedAllocation{
			allocPinned("A", 2, 2),
			alloc("B", 8, 2),
		},
		req:              orderedRequest{size: 8},
		budget:           orderedBudgetOptimal,
		wantPlacement:    4,
		wantEvictionIDs:  []string{"B"},
		wantEvictNewSets: map[string]int{"B": 0},
	})
}

func TestOrderedAllocator_PinnedBlocksAllPlacements(t *testing.T) {
	// A pinned at 4..5; req size=8.  Must avoid 4..5; impossible (any 8-cell
	// run includes either 4 or 5 since chain is 12).
	runSchedulerTestCase(t, schedulerTestCase{
		name:        "pinned splits chain such that no 8-cell run avoids it",
		chainSize:   12,
		bound:       []orderedAllocation{allocPinned("A", 4, 2)},
		req:         orderedRequest{size: 8},
		budget:      orderedBudgetOptimal,
		wantPending: true,
	})
}

// ---------------------------------------------------------------------------
// Section 5: Priority — same-or-lower priority can be evicted; higher cannot.
// ---------------------------------------------------------------------------

func TestOrderedAllocator_HigherPriorityNotEvictable(t *testing.T) {
	// A at 4..5 with prio=200; req size=8 prio=150 → A is higher priority,
	// effectively pinned for this request. No 8-cell run avoids A. Pending.
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "higher-prio bound is not evicted by lower-prio req",
		chainSize: 12,
		bound: []orderedAllocation{
			allocPrio("A", 4, 2, 200),
		},
		req:         orderedRequest{size: 8, priority: 150},
		budget:      orderedBudgetOptimal,
		wantPending: true,
	})
}

func TestOrderedAllocator_EqualPriorityEvictable(t *testing.T) {
	// A at 4..5 prio=150; req size=8 prio=150 → equal priority, A can be
	// moved.  Same shape as TestOrderedAllocator_CompactSingleEvict but
	// with explicit priorities.  Leftmost-placement tiebreaker: req@0..7
	// with A→8.
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "equal priority can be evicted",
		chainSize: 12,
		bound: []orderedAllocation{
			allocPrio("A", 4, 2, 150),
		},
		req:              orderedRequest{size: 8, priority: 150},
		budget:           orderedBudgetOptimal,
		wantPlacement:    0,
		wantEvictionIDs:  []string{"A"},
		wantEvictNewSets: map[string]int{"A": 8},
	})
}

func TestOrderedAllocator_PriorityPreservation(t *testing.T) {
	// A=4@0..3 prio=200, B=4@4..7 prio=100, idle 8..11; C size=8 prio=150.
	// First-fit on existing layout: only 8..11 (size 4) free, no 8-cell.
	// Must evict to fit. A is higher than C — pinned. B is lower — movable.
	// Try start=0..7: contains A → reject. start=4..11: contains B (4..7),
	// must evict B. Place req at 4..11. Free for B = [0..3]? A is at 0..3.
	// No room for B. Sum: 4+4+8 = 16 > 12. Infeasible.  Verify Pending.
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "priority preservation but capacity-infeasible",
		chainSize: 12,
		bound: []orderedAllocation{
			allocPrio("A", 0, 4, 200),
			allocPrio("B", 4, 4, 100),
		},
		req:         orderedRequest{size: 8, priority: 150},
		budget:      orderedBudgetOptimal,
		wantPending: true,
	})
}

func TestOrderedAllocator_PriorityFeasibleCompaction(t *testing.T) {
	// A=2@0..1 prio=200, B=2@4..5 prio=100, idle elsewhere; C size=6 prio=150.
	// First-fit: free runs are [2..3]=2 and [6..11]=6. Placing at 6..11 fits
	// without compaction.
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "first-fit takes the right gap",
		chainSize: 12,
		bound: []orderedAllocation{
			allocPrio("A", 0, 2, 200),
			allocPrio("B", 4, 2, 100),
		},
		req:           orderedRequest{size: 6, priority: 150},
		budget:        orderedBudgetOptimal,
		wantPlacement: 6,
	})
}

// ---------------------------------------------------------------------------
// Section 6: Memory-allocator-style fragmentation patterns.
// ---------------------------------------------------------------------------

// checkerboard returns a layout with size-1 evictable allocations at every
// even index, simulating the classic "checkerboard fragmentation" stress test.
func checkerboard(chainSize int) []orderedAllocation {
	result := make([]orderedAllocation, 0, chainSize/2)
	for i := 0; i < chainSize; i += 2 {
		result = append(result, alloc(fmt.Sprintf("c%d", i), i, 1))
	}
	return result
}

func TestOrderedAllocator_Checkerboard_BudgetZero(t *testing.T) {
	// chain=12, bounds at 0,2,4,6,8,10 (each size 1).  Free runs are size 1
	// each. Any req size ≥ 2 stays pending at budget=0.
	runSchedulerTestCase(t, schedulerTestCase{
		name:        "checkerboard, budget=0, req size=2 → pending",
		chainSize:   12,
		bound:       checkerboard(12),
		req:         orderedRequest{size: 2},
		budget:      orderedBudgetNoCompact,
		wantPending: true,
	})
}

func TestOrderedAllocator_Checkerboard_BudgetCanCompact(t *testing.T) {
	// chain=12, 6 size-1 evictables at 0,2,4,6,8,10. req=6 with budget=5.
	// The optimal plan evicts the 3 leftmost bounds (c0, c2, c4), placing
	// the request at 0..5 and re-placing the 3 evictees in the trailing
	// gaps at 7, 9, 11. Three evictions is well within the budget of 5.
	// Memory-allocator analog: a "stop-and-copy" half-collection that
	// frees a contiguous region from the leftmost half.
	a := &orderedAllocator{chainSize: 12, bound: checkerboard(12)}
	plan := a.schedule(orderedRequest{size: 6}, 5)
	if plan.pending {
		t.Fatalf("unexpected pending: %s", plan.pendingReason)
	}
	if len(plan.evictions) != 3 {
		t.Errorf("evictions = %d, want 3 (optimal)", len(plan.evictions))
	}
	if plan.placement != 0 {
		t.Errorf("placement = %d, want 0 (leftmost)", plan.placement)
	}
	checkPlanInvariants(t, 12, checkerboard(12), orderedRequest{size: 6}, plan)
}

func TestOrderedAllocator_Checkerboard_CapacityInfeasible(t *testing.T) {
	// chain=12, 6 size-1 evictables totalling 6 cells used.  A request
	// for size=7 cannot ever fit: total demand = 7 + 6 = 13 > 12 = chain
	// capacity.  Verify pending at every budget.  This is a classic
	// memory-allocator "OOM" scenario — no compaction is enough; the
	// caller would have to evict-and-not-relocate (i.e. preempt) some
	// workloads, which our allocator (intentionally) does not do.
	a := &orderedAllocator{chainSize: 12, bound: checkerboard(12)}
	for _, budget := range []int{0, 1, 3, 6, 10, orderedBudgetOptimal} {
		plan := a.schedule(orderedRequest{size: 7}, budget)
		if !plan.pending {
			t.Errorf("budget=%d: expected pending (capacity-infeasible), got placement=%d evictions=%v",
				budget, plan.placement, plan.evictions)
		}
	}
}

func TestOrderedAllocator_WorstCaseFragmentation(t *testing.T) {
	// Pack many small bounds with single-cell gaps. Verify optimal compaction
	// finds a valid plan whenever capacity permits.
	chainSize := 16
	bounds := []orderedAllocation{}
	// Layout: gap, A=1, gap, B=2, gap, C=1, gap, D=2, gap, E=1, gap.
	// positions:  0   1    2   3..4  5    6    7   8..9  10   11   12-15
	bounds = append(bounds,
		alloc("A", 1, 1),
		alloc("B", 3, 2),
		alloc("C", 6, 1),
		alloc("D", 8, 2),
		alloc("E", 11, 1),
	)
	totalBound := 0
	for _, b := range bounds {
		totalBound += b.size
	}
	// req=8: bound total 7, chain 16, free = 9. Need 8 contiguous. Heuristic
	// says: compact bounds leftward (positions 0..6), free 7..15 = 9 cells.
	a := &orderedAllocator{chainSize: chainSize, bound: bounds}
	plan := a.schedule(orderedRequest{size: 8}, orderedBudgetOptimal)
	if plan.pending {
		t.Fatalf("expected fit, got pending: %s", plan.pendingReason)
	}
	checkPlanInvariants(t, chainSize, bounds, orderedRequest{size: 8}, plan)
}

// ---------------------------------------------------------------------------
// Section 7: Optimal-vs-budget parity.
// For every input where a budget-bounded plan exists, the optimal solver
// must produce a plan with eviction count <= the budget plan's count.
// ---------------------------------------------------------------------------

func TestOrderedAllocator_OptimalNeverWorse(t *testing.T) {
	scenarios := []struct {
		name      string
		chainSize int
		bound     []orderedAllocation
		req       orderedRequest
	}{
		{
			"checkerboard12 req=4",
			12, checkerboard(12), orderedRequest{size: 4},
		},
		{
			"checkerboard16 req=10",
			16, checkerboard(16), orderedRequest{size: 10},
		},
		{
			"compact single",
			12,
			[]orderedAllocation{alloc("B", 4, 2)},
			orderedRequest{size: 8},
		},
		{
			"two evictables, narrow gap",
			16,
			[]orderedAllocation{
				alloc("A", 2, 2),
				alloc("B", 6, 4),
				alloc("C", 12, 2),
			},
			orderedRequest{size: 10},
		},
	}
	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			a := &orderedAllocator{chainSize: sc.chainSize, bound: sc.bound}
			optimalPlan := a.schedule(sc.req, orderedBudgetOptimal)
			if optimalPlan.pending {
				return // nothing further to check; optimal said no fit
			}
			optimalCount := len(optimalPlan.evictions)
			// For every budget >= optimalCount, a plan should exist with
			// evictions <= optimalCount.
			for budget := optimalCount; budget <= optimalCount+3; budget++ {
				plan := a.schedule(sc.req, budget)
				if plan.pending {
					t.Errorf("budget=%d went pending while optimal succeeded with %d evictions",
						budget, optimalCount)
					continue
				}
				if len(plan.evictions) > budget {
					t.Errorf("budget=%d got %d evictions (exceeds budget)",
						budget, len(plan.evictions))
				}
				if len(plan.evictions) > optimalCount {
					t.Errorf("budget=%d got %d evictions, optimal had %d (heuristic worse than optimal)",
						budget, len(plan.evictions), optimalCount)
				}
			}
			// For budget < optimalCount, plans should be either pending or
			// truly cheaper (which contradicts "optimal" — i.e. should not
			// happen).
			for budget := 0; budget < optimalCount; budget++ {
				plan := a.schedule(sc.req, budget)
				if !plan.pending && len(plan.evictions) < optimalCount {
					t.Errorf("budget=%d found cheaper plan (%d) than 'optimal' (%d) — solver bug",
						budget, len(plan.evictions), optimalCount)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Section 7b: Disruption-budget knob spectrum.
// Demonstrates the budget knob's full range (0 → ∞) on a single fixed
// scenario: at budget=0 the request is pending; at low budgets it remains
// pending until the optimum is reachable; once the budget hits the optimal
// eviction count, the plan stabilises (more budget never produces *more*
// evictions, since the score function prefers fewer).
// ---------------------------------------------------------------------------

func TestOrderedAllocator_BudgetSpectrum(t *testing.T) {
	// chain=12, three size-2 evictable bounds at 2..3, 5..6, 8..9, idle
	// elsewhere; req size=6.  Free-runs initially are [0..1]=2, [4..4]=1,
	// [7..7]=1, [10..11]=2 — no 6-run, so first-fit fails.
	//
	// The optimum requires evicting two of the three (e.g. A+B → req @ 0..5,
	// A→6, B→10; or B+C → req @ 4..9 with A staying, B→0, C→10) for a
	// total of 2 evictions.  No 1-eviction plan exists because removing
	// any single bound still leaves max-run 5.  So:
	//   budget = 0 → pending (no compaction allowed)
	//   budget = 1 → pending (cap below optimum)
	//   budget = 2 → 2-eviction plan
	//   budget = 3 → 2-eviction plan (extra headroom unused)
	//   budget = 100 → 2-eviction plan (extra headroom unused)
	//   budget = -1 (optimal) → 2-eviction plan (same shape)
	bounds := []orderedAllocation{
		alloc("A", 2, 2),
		alloc("B", 5, 2),
		alloc("C", 8, 2),
	}
	req := orderedRequest{size: 6}
	a := &orderedAllocator{chainSize: 12, bound: bounds}

	cases := []struct {
		budget         int
		wantPending    bool
		wantEvictCount int
	}{
		{budget: 0, wantPending: true},
		{budget: 1, wantPending: true},
		{budget: 2, wantEvictCount: 2},
		{budget: 3, wantEvictCount: 2},
		{budget: 100, wantEvictCount: 2},
		{budget: orderedBudgetOptimal, wantEvictCount: 2},
	}
	for _, c := range cases {
		name := fmt.Sprintf("budget=%d", c.budget)
		t.Run(name, func(t *testing.T) {
			plan := a.schedule(req, c.budget)
			if plan.pending != c.wantPending {
				t.Fatalf("pending = %v (reason=%q), want %v",
					plan.pending, plan.pendingReason, c.wantPending)
			}
			if c.wantPending {
				return
			}
			if len(plan.evictions) != c.wantEvictCount {
				t.Errorf("evictions = %d, want %d", len(plan.evictions), c.wantEvictCount)
			}
			if len(plan.evictions) > c.budget && c.budget != orderedBudgetOptimal {
				t.Errorf("plan has %d evictions, exceeds budget %d",
					len(plan.evictions), c.budget)
			}
			checkPlanInvariants(t, 12, bounds, req, plan)
		})
	}
}

// TestOrderedAllocator_BudgetMonotonicity asserts a structural property of
// the knob: increasing the budget never makes the result strictly worse
// (more evictions, higher placement) on the same input.  This is the formal
// statement of "infinity ≥ all finite budgets".
func TestOrderedAllocator_BudgetMonotonicity(t *testing.T) {
	scenarios := []struct {
		name      string
		chainSize int
		bound     []orderedAllocation
		req       orderedRequest
	}{
		{
			"checkerboard 12, req=6",
			12, checkerboard(12), orderedRequest{size: 6},
		},
		{
			"three small evictables, req=6",
			12,
			[]orderedAllocation{
				alloc("A", 2, 2), alloc("B", 5, 2), alloc("C", 8, 2),
			},
			orderedRequest{size: 6},
		},
		{
			"single-bound mid-chain, req=8",
			12,
			[]orderedAllocation{alloc("X", 4, 2)},
			orderedRequest{size: 8},
		},
	}
	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			a := &orderedAllocator{chainSize: sc.chainSize, bound: sc.bound}
			budgets := []int{0, 1, 2, 3, 5, 10, 100, orderedBudgetOptimal}
			lastEvictCount := -1
			lastPending := true
			for _, b := range budgets {
				plan := a.schedule(sc.req, b)
				// Pending may flip from true to false as budget grows, but
				// must never flip back (budget=∞ implies anything finite
				// could also reach that plan, so once feasible, always
				// feasible at higher budgets).
				if !lastPending && plan.pending {
					t.Errorf("budget %d went pending after an earlier feasible budget", b)
				}
				// Eviction count must be non-increasing as budget grows
				// past the optimal point: extra headroom never adds evictions.
				if !plan.pending {
					if lastEvictCount != -1 && len(plan.evictions) > lastEvictCount && b != orderedBudgetOptimal {
						t.Errorf("budget %d produced %d evictions, more than smaller budget's %d",
							b, len(plan.evictions), lastEvictCount)
					}
					lastEvictCount = len(plan.evictions)
					lastPending = false
					checkPlanInvariants(t, sc.chainSize, sc.bound, sc.req, plan)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Section 8: Boundary conditions.
// ---------------------------------------------------------------------------

func TestOrderedAllocator_FullSizeRequest(t *testing.T) {
	// Empty chain, request size = chainSize → place at 0.
	runSchedulerTestCase(t, schedulerTestCase{
		name:          "request equals chain size",
		chainSize:     12,
		req:           orderedRequest{size: 12},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 0,
	})
}

func TestOrderedAllocator_FullSizeRequest_OneEvictBlocks(t *testing.T) {
	// One bound of size 1 at index 6. req size=12 (whole chain) at any
	// budget: must evict to fit a 12-run. Total bound + req = 13 > 12.
	// Infeasible at any budget.
	runSchedulerTestCase(t, schedulerTestCase{
		name:        "request equals chain size, one bound, infeasible",
		chainSize:   12,
		bound:       []orderedAllocation{alloc("X", 6, 1)},
		req:         orderedRequest{size: 12},
		budget:      orderedBudgetOptimal,
		wantPending: true,
	})
}

func TestOrderedAllocator_TightFit(t *testing.T) {
	// Bounds at 0..3 and 8..11, req size=4 → fits at 4..7 (the only gap).
	runSchedulerTestCase(t, schedulerTestCase{
		name:      "tight fit in middle gap",
		chainSize: 12,
		bound: []orderedAllocation{
			alloc("A", 0, 4),
			alloc("B", 8, 4),
		},
		req:           orderedRequest{size: 4},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 4,
	})
}

// ---------------------------------------------------------------------------
// Section 9: Tie-breaking (deterministic, by lowest-priority + earliest
// boundAt + lex id).
// ---------------------------------------------------------------------------

func TestOrderedAllocator_TieBreakLowerPriority(t *testing.T) {
	// Two evictables, both size 2, same start delta.  Eviction must happen.
	// Higher-prio evictable should remain in place vs lower-prio.
	a := &orderedAllocator{
		chainSize: 12,
		bound: []orderedAllocation{
			allocPrio("hi", 4, 2, 100),
			allocPrio("lo", 6, 2, 50),
			alloc("X", 0, 4), // pinned-by-position pre-existing
			alloc("Y", 8, 4),
		},
	}
	a.bound[2].evictable = false
	a.bound[3].evictable = false
	// Request size=4 prio=200, budget=2.  No first-fit (free is 4..7=4 actually wait).
	// X@0..3, hi@4..5, lo@6..7, Y@8..11.  Free = nothing.  Need to evict.
	// Must place req in some 4-run.  The only candidate slots are where
	// hi+lo currently sit: 4..7. Evict both. Re-place hi+lo where? No room.
	// Sum = 4+2+2+4+4 = 16 > 12. Infeasible.  Pending.
	plan := a.schedule(orderedRequest{size: 4, priority: 200}, 2)
	if !plan.pending {
		t.Errorf("expected pending (capacity-infeasible), got placement=%d", plan.placement)
	}
}

func TestOrderedAllocator_TieBreakDeterministic(t *testing.T) {
	// chain=14, A=2@4..5, B=2@8..9, idle elsewhere; req=10, budget=2.
	// Sum of allocations + req = 14 = chainSize (tight fit).  The unique
	// plan must evict both A and B and re-place them.  The placement
	// tie-break (leftmost) picks req@0..9; A and B re-place into the
	// 4-cell trailing gap [10..13]: A→10, B→12.  The plan's eviction
	// list is sorted by newStart asc, then id asc, so it must be
	// [(A,10),(B,12)] deterministically.
	a := &orderedAllocator{
		chainSize: 14,
		bound: []orderedAllocation{
			alloc("A", 4, 2),
			alloc("B", 8, 2),
		},
	}
	plan := a.schedule(orderedRequest{size: 10}, 2)
	if plan.pending {
		t.Fatalf("unexpected pending: %s", plan.pendingReason)
	}
	if plan.placement != 0 {
		t.Errorf("placement = %d, want 0 (leftmost)", plan.placement)
	}
	if len(plan.evictions) != 2 {
		t.Fatalf("evictions = %d, want 2", len(plan.evictions))
	}
	got := []string{plan.evictions[0].id, plan.evictions[1].id}
	want := []string{"A", "B"}
	if !slices.Equal(got, want) {
		t.Errorf("eviction order = %v, want %v", got, want)
	}
	if plan.evictions[0].newStart != 10 || plan.evictions[1].newStart != 12 {
		t.Errorf("eviction newStarts = %d,%d; want 10,12",
			plan.evictions[0].newStart, plan.evictions[1].newStart)
	}
	checkPlanInvariants(t, a.chainSize, a.bound, orderedRequest{size: 10}, plan)
}

// ---------------------------------------------------------------------------
// Section 10: Stress / churn — random alloc-and-free sequence; invariants
// hold throughout.
// ---------------------------------------------------------------------------

func TestOrderedAllocator_RandomChurn(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	chainSize := 16
	live := []orderedAllocation{}
	nextID := 0

	for step := 0; step < 200; step++ {
		// Random op: place / remove.
		if len(live) > 0 && rng.Intn(2) == 0 {
			// remove a random live one
			idx := rng.Intn(len(live))
			live = append(live[:idx], live[idx+1:]...)
			continue
		}
		// place
		size := 1 + rng.Intn(4)
		req := orderedRequest{size: size}
		budget := rng.Intn(4) // 0..3
		a := &orderedAllocator{chainSize: chainSize, bound: live}
		plan := a.schedule(req, budget)
		checkPlanInvariants(t, chainSize, live, req, plan)
		if plan.pending {
			continue
		}
		// Apply the plan: relocate evicted, then add the new allocation.
		newLayout := make([]orderedAllocation, 0, len(live)+1)
		moved := make(map[string]int, len(plan.evictions))
		for _, e := range plan.evictions {
			moved[e.id] = e.newStart
		}
		for _, b := range live {
			if ns, ok := moved[b.id]; ok {
				b.start = ns
			}
			newLayout = append(newLayout, b)
		}
		newLayout = append(newLayout, alloc(fmt.Sprintf("R%d", nextID), plan.placement, size))
		nextID++
		live = newLayout
		// Validate no overlaps in the final layout.
		validateLayout(t, chainSize, live)
	}
}

func validateLayout(t *testing.T, chainSize int, bound []orderedAllocation) {
	t.Helper()
	occupancy := make(map[int]string, chainSize)
	for _, b := range bound {
		if b.start < 0 || b.end() > chainSize {
			t.Errorf("layout out of bounds: %+v (chainSize=%d)", b, chainSize)
			continue
		}
		for i := b.start; i < b.end(); i++ {
			if existing, found := occupancy[i]; found {
				t.Errorf("overlap at %d between %q and %q", i, existing, b.id)
			}
			occupancy[i] = b.id
		}
	}
}

// ---------------------------------------------------------------------------
// Section 10b: Victim-comparator semantics — lex-max, age, pluggable.
// ---------------------------------------------------------------------------

// TestComparePrecomputed_LexMax verifies the plan-level comparator's
// lex-max behavior in isolation, without the geometric complexity of
// constructing an end-to-end scenario. The comparator's contract is:
//
//  1. Fewer evictions wins.
//  2. Among same-count plans, sort each plan's victims descending by
//     victimLess (most expensive first) and lex-compare. Plan whose
//     most-expensive-victim is cheaper wins; ties on next-most-expensive,
//     etc. This is the formal "minimum maximum priority" criterion that
//     beats sum-of-priorities on cases like [50,50] vs [80,10].
//  3. Among lex-tied plans, smaller relocation distance wins.
//  4. Among all-tied plans, leftmost placement.
func TestComparePrecomputed_LexMax(t *testing.T) {
	a := &orderedAllocator{}

	// Helper: build a sortedVictims list from priorities, with ids derived
	// from priority for determinism.  The list is pre-sorted descending
	// by victimLess, as comparePrecomputed expects.
	mk := func(prios ...int32) []orderedAllocation {
		out := make([]orderedAllocation, len(prios))
		for i, p := range prios {
			out[i] = orderedAllocation{id: fmt.Sprintf("p%d", p), priority: p, boundAt: 100}
		}
		sort.Slice(out, func(i, j int) bool {
			return defaultVictimLess(out[i], out[j]) > 0
		})
		return out
	}
	plan := func(placement int, n int) orderedPlan {
		evs := make([]orderedEviction, n)
		return orderedPlan{placement: placement, evictions: evs}
	}

	cases := []struct {
		name        string
		aPlan       orderedPlan
		aVictims    []orderedAllocation
		aDist       int
		bPlan       orderedPlan
		bVictims    []orderedAllocation
		bDist       int
		wantNegLess bool // true if a < b
	}{
		{
			name:  "fewer evictions wins (1 vs 2)",
			aPlan: plan(0, 1), aVictims: mk(100), aDist: 0,
			bPlan: plan(0, 2), bVictims: mk(10, 10), bDist: 0,
			wantNegLess: true,
		},
		{
			name: "lex-max: [50,50] beats [80,10] (50 < 80 at position 0)",
			// Sum says [80,10]=90 < [50,50]=100, but lex says [50,50] wins.
			aPlan: plan(0, 2), aVictims: mk(50, 50), aDist: 0,
			bPlan: plan(0, 2), bVictims: mk(80, 10), bDist: 0,
			wantNegLess: true,
		},
		{
			name: "lex-max ties at position 0, breaks at position 1",
			// Both have max=80. Second-max is 10 vs 50. [80,10] wins.
			aPlan: plan(0, 2), aVictims: mk(80, 10), aDist: 0,
			bPlan: plan(0, 2), bVictims: mk(80, 50), bDist: 0,
			wantNegLess: true,
		},
		{
			name:  "lex-tied: smaller relocation distance wins",
			aPlan: plan(0, 2), aVictims: mk(50, 50), aDist: 5,
			bPlan: plan(0, 2), bVictims: mk(50, 50), bDist: 9,
			wantNegLess: true,
		},
		{
			name:  "all-tied: leftmost placement wins",
			aPlan: plan(2, 2), aVictims: mk(50, 50), aDist: 0,
			bPlan: plan(8, 2), bVictims: mk(50, 50), bDist: 0,
			wantNegLess: true,
		},
		{
			name: "sum-vs-lex divergence: [50,50] beats [10,90] under lex",
			// Sum: [10,90]=100 < [50,50]=100 (tied). Lex: pos0 50<90.
			// Lex picks [50,50].
			aPlan: plan(0, 2), aVictims: mk(50, 50), aDist: 0,
			bPlan: plan(0, 2), bVictims: mk(90, 10), bDist: 0,
			wantNegLess: true,
		},
		{
			name: "fewer evictions still wins even with worse priorities",
			// [200] (sum 200) beats [10,10,10] (sum 30) on count alone.
			aPlan: plan(0, 1), aVictims: mk(200), aDist: 0,
			bPlan: plan(0, 3), bVictims: mk(10, 10, 10), bDist: 0,
			wantNegLess: true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := a.comparePrecomputed(c.aPlan, c.aVictims, c.aDist, 0, c.bPlan, c.bVictims, c.bDist, 0)
			if c.wantNegLess && got >= 0 {
				t.Errorf("compare(a, b) = %d, want negative (a should win)", got)
			}
			// Antisymmetry: compare(b, a) should be opposite-signed.
			rev := a.comparePrecomputed(c.bPlan, c.bVictims, c.bDist, 0, c.aPlan, c.aVictims, c.aDist, 0)
			if (got > 0) == (rev > 0) && got != 0 {
				t.Errorf("antisymmetry violated: compare(a,b)=%d compare(b,a)=%d", got, rev)
			}
		})
	}
}

// TestOrderedAllocator_AgeTiebreaker verifies that among equal-priority
// candidates, the older workload is protected. Age is the secondary
// tiebreaker in the default victim comparator: newer workloads (higher
// boundAt) are cheaper to evict because they've invested less in setup.
func TestOrderedAllocator_AgeTiebreaker(t *testing.T) {
	older := orderedAllocation{
		id: "older", start: 4, size: 2, priority: 50, boundAt: 100, evictable: true,
	}
	newer := orderedAllocation{
		id: "newer", start: 8, size: 2, priority: 50, boundAt: 200, evictable: true,
	}
	a := &orderedAllocator{
		chainSize: 12,
		bound:     []orderedAllocation{older, newer},
	}
	// Request priority must be ≥ victim priority for either to be evictable.
	plan := a.schedule(orderedRequest{size: 6, priority: 100}, orderedBudgetOptimal)
	if plan.pending {
		t.Fatalf("unexpected pending: %s", plan.pendingReason)
	}
	// Both 1-eviction plans are feasible (evict either older or newer).
	// The age tiebreaker should pick the newer one as the victim.
	if len(plan.evictions) != 1 {
		t.Fatalf("evictions = %d, want 1", len(plan.evictions))
	}
	if plan.evictions[0].id != "newer" {
		t.Errorf("evicted %q, want %q (older workload should be protected)",
			plan.evictions[0].id, "newer")
	}
	checkPlanInvariants(t, 12, a.bound, orderedRequest{size: 6, priority: 100}, plan)
}

// TestOrderedAllocator_PluggableComparator demonstrates that supplying a
// custom victimLess overrides the default ordering.  Same layout, two
// runs with opposite comparators, opposite outcomes.  The integration
// will pass in a comparator that delegates to Kueue's CandidatesOrdering
// so chain compaction stays consistent with cohort-wide preemption
// decisions.
func TestOrderedAllocator_PluggableComparator(t *testing.T) {
	bounds := []orderedAllocation{
		allocPrio("A", 4, 2, 50),
		allocPrio("B", 8, 2, 100),
	}
	req := orderedRequest{size: 6, priority: 200}

	// Default: lower priority is cheaper to evict → pick A.
	a1 := &orderedAllocator{chainSize: 12, bound: bounds}
	plan1 := a1.schedule(req, orderedBudgetOptimal)
	if plan1.pending {
		t.Fatalf("default unexpectedly pending: %s", plan1.pendingReason)
	}
	if len(plan1.evictions) != 1 || plan1.evictions[0].id != "A" {
		t.Errorf("default evicted %v, want [A]", plan1.evictions)
	}

	// Custom: higher priority is cheaper to evict (e.g. an admin policy that
	// rotates whichever workload has accumulated the most quota lately).
	inverted := func(x, y orderedAllocation) int {
		if c := cmp.Compare(y.priority, x.priority); c != 0 {
			return c
		}
		return cmp.Compare(x.id, y.id)
	}
	a2 := &orderedAllocator{
		chainSize:  12,
		bound:      bounds,
		victimLess: inverted,
	}
	plan2 := a2.schedule(req, orderedBudgetOptimal)
	if plan2.pending {
		t.Fatalf("inverted unexpectedly pending: %s", plan2.pendingReason)
	}
	if len(plan2.evictions) != 1 || plan2.evictions[0].id != "B" {
		t.Errorf("inverted evicted %v, want [B]", plan2.evictions)
	}
}

// TestDefaultVictimLess unit-tests the comparator function in isolation.
func TestDefaultVictimLess(t *testing.T) {
	cases := []struct {
		name string
		a, b orderedAllocation
		want int // negative, zero, or positive
	}{
		{
			name: "lower priority cheaper",
			a:    orderedAllocation{id: "a", priority: 10, boundAt: 100},
			b:    orderedAllocation{id: "b", priority: 100, boundAt: 100},
			want: -1,
		},
		{
			name: "newer cheaper at equal priority",
			a:    orderedAllocation{id: "a", priority: 50, boundAt: 200},
			b:    orderedAllocation{id: "b", priority: 50, boundAt: 100},
			want: -1,
		},
		{
			name: "id tiebreak when priority and age tie",
			a:    orderedAllocation{id: "a", priority: 50, boundAt: 100},
			b:    orderedAllocation{id: "b", priority: 50, boundAt: 100},
			want: -1,
		},
		{
			name: "fully equal",
			a:    orderedAllocation{id: "x", priority: 50, boundAt: 100},
			b:    orderedAllocation{id: "x", priority: 50, boundAt: 100},
			want: 0,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := defaultVictimLess(c.a, c.b)
			switch {
			case c.want < 0 && got >= 0:
				t.Errorf("got %d, want negative", got)
			case c.want > 0 && got <= 0:
				t.Errorf("got %d, want positive", got)
			case c.want == 0 && got != 0:
				t.Errorf("got %d, want zero", got)
			}
			// Antisymmetry (when not equal): less(a,b) and less(b,a) have
			// opposite signs.
			rev := defaultVictimLess(c.b, c.a)
			if c.want != 0 && (got > 0) == (rev > 0) {
				t.Errorf("antisymmetry violated: less(a,b)=%d less(b,a)=%d", got, rev)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Section 10c: Real-cluster scenario — LWS evictable, JobSet not.
//
// Models the user's 12-node Thunderbolt daisy-chain.  Per-kind defaults
// from the integration plan:
//
//	LeaderWorkerSet  → evictable=true  (inference; pods recreate
//	                                    atomically via maxUnavailable=100%)
//	JobSet           → evictable=false (batch; eviction loses progress)
//	Job              → evictable=false (same as JobSet)
//
// The pure allocator only sees the `evictable` flag.  The same geometry
// is exercised three times — all-LWS, all-JobSet, mixed — so flipping
// the flag changes whether a fragmented incoming request gets admitted.
// This is how the kind→evictable mapping shows up at the allocator layer.
// ---------------------------------------------------------------------------

func TestOrderedAllocator_LWSvsJobSetEvictability(t *testing.T) {
	const chainSize = 12
	req := orderedRequest{size: 6}
	const budget = 5

	// Geometry: two existing size-3 workloads at chain-index 0..2 and 6..8.
	// Free runs are [3..5]=3 and [9..11]=3 — neither fits a 6-cell request,
	// so first-fit always fails and the allocator must consider compaction.

	cases := []struct {
		name          string
		bound         []orderedAllocation
		wantPending   bool
		wantEvictedID string // populated only when admission expected
		wantPlacement int    // populated only when admission expected
		wantNewStart  int    // newStart of the single evicted bound
	}{
		{
			name: "two LWSes coexist — admit new LWS by moving the newer one",
			bound: []orderedAllocation{
				{id: "lws-A", start: 0, size: 3, evictable: true, boundAt: 100},
				{id: "lws-B", start: 6, size: 3, evictable: true, boundAt: 200},
			},
			wantPending: false,
			// Default victim comparator: equal priority → newer (higher
			// boundAt) is cheaper to evict.  lws-B is newer, so it moves.
			wantEvictedID: "lws-B",
			// Subset {lws-B}: staying = lws-A@0..2. FFL gives free run
			// [3..11]=9, request placed leftmost at 3..8.  lws-B re-placed
			// at 9..11 (the only remaining 3-cell interval).
			wantPlacement: 3,
			wantNewStart:  9,
		},
		{
			name: "two JobSets coexist — request stays pending",
			bound: []orderedAllocation{
				{id: "jobset-A", start: 0, size: 3, evictable: false, boundAt: 100},
				{id: "jobset-B", start: 6, size: 3, evictable: false, boundAt: 200},
			},
			wantPending: true,
		},
		{
			name: "JobSet pins indices — LWS routes around it",
			bound: []orderedAllocation{
				{id: "lws-A", start: 0, size: 3, evictable: true, boundAt: 100},
				{id: "jobset-B", start: 6, size: 3, evictable: false, boundAt: 200},
			},
			wantPending: false,
			// Only lws-A is evictable.  Subset {lws-A}: staying =
			// jobset-B@6..8. FFL gives free run [0..5]=6 (left of JobSet),
			// request placed leftmost at 0..5. lws-A re-placed at 9..11.
			wantEvictedID: "lws-A",
			wantPlacement: 0,
			wantNewStart:  9,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := &orderedAllocator{chainSize: chainSize, bound: c.bound}
			plan := a.schedule(req, budget)

			if plan.pending != c.wantPending {
				t.Fatalf("pending=%v (reason=%q), want %v",
					plan.pending, plan.pendingReason, c.wantPending)
			}
			if c.wantPending {
				return
			}
			if plan.placement != c.wantPlacement {
				t.Errorf("placement=%d, want %d", plan.placement, c.wantPlacement)
			}
			if len(plan.evictions) != 1 {
				t.Fatalf("evictions=%d, want 1", len(plan.evictions))
			}
			if plan.evictions[0].id != c.wantEvictedID {
				t.Errorf("evicted=%q, want %q", plan.evictions[0].id, c.wantEvictedID)
			}
			if plan.evictions[0].newStart != c.wantNewStart {
				t.Errorf("evicted newStart=%d, want %d",
					plan.evictions[0].newStart, c.wantNewStart)
			}
			checkPlanInvariants(t, chainSize, c.bound, req, plan)
		})
	}
}

// TestOrderedAllocator_JobSetBlocksMidChain exercises a darker variant of
// the mixed case: a JobSet sits in the middle of the chain (indices 4..7),
// fragmenting the chain such that no contiguous 6-run can ever exist
// while the JobSet stays put. Compaction with any budget must give up.
// This shows non-evictable bounds genuinely partition the chain rather
// than just costing budget.
func TestOrderedAllocator_JobSetBlocksMidChain(t *testing.T) {
	const chainSize = 12
	req := orderedRequest{size: 6}
	bound := []orderedAllocation{
		{id: "lws-A", start: 0, size: 3, evictable: true, boundAt: 100},
		{id: "jobset-mid", start: 4, size: 4, evictable: false, boundAt: 50},
	}
	a := &orderedAllocator{chainSize: chainSize, bound: bound}
	for _, budget := range []int{0, 1, 5, orderedBudgetOptimal} {
		t.Run(fmt.Sprintf("budget=%d", budget), func(t *testing.T) {
			plan := a.schedule(req, budget)
			if !plan.pending {
				t.Errorf("expected pending (mid-chain JobSet partitions chain into "+
					"max-3-cell free runs), got placement=%d evictions=%v",
					plan.placement, plan.evictions)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Section 11: Cache warmth — re-placement prefers previous positions.
//
// Weights and compile caches are node-local; a workload re-placed onto
// nodes it previously occupied skips a ~10 minute cold reload. The
// preference is a scoring term, never a hard constraint.
// ---------------------------------------------------------------------------

// TestOrderedAllocator_WarmthPrefersPreviousPositions gives the evicted
// victim two feasible re-placements: run {8,9} (closer to its recorded
// start, so pure relocation-distance would pick it) and run {12,13}
// (overlapping position 12, which the victim actually occupies). The
// warm run must win.
//
// Geometry (chain=14):
//
//	P pinned at {10,11};
//	V evictable, size 2, fragmented across positions {3,12};
//	request size 8, budget 1.
//
// Only V's eviction can produce an 8-run. After the request takes [0,8),
// the free intervals are {8,9} and {12,13}. Overlap with V's previous
// positions {3,12}: run {8,9} → 0, run {12,13} → 1. Warmth picks 12.
func TestOrderedAllocator_WarmthPrefersPreviousPositions(t *testing.T) {
	bound := []orderedAllocation{
		{id: "P", start: 10, size: 2, evictable: false, boundAt: 50},
		{id: "V", start: 3, size: 2, positions: []int{3, 12}, evictable: true, boundAt: 100},
	}
	a := &orderedAllocator{chainSize: 14, bound: bound}
	plan := a.schedule(orderedRequest{size: 8}, 1)
	if plan.pending {
		t.Fatalf("unexpected pending: %s", plan.pendingReason)
	}
	if plan.placement != 0 {
		t.Errorf("placement = %d, want 0 (leftmost feasible)", plan.placement)
	}
	if len(plan.evictions) != 1 || plan.evictions[0].id != "V" {
		t.Fatalf("evictions = %+v, want exactly [V]", plan.evictions)
	}
	if got := plan.evictions[0].newStart; got != 12 {
		t.Errorf("V newStart = %d, want 12 (warm run {12,13} overlaps previous position 12; "+
			"run {8,9} is closer to the recorded start but cold)", got)
	}
}

// TestOrderedAllocator_FragmentedIncidentGeometry is the allocator-level
// statement of the 2026-07-10 incident: a 2-slice workload fragmented at
// {0,2} on a 12-cell chain, a 10-cell request, budget 1. The plan must
// evict the fragmented workload and re-place it on its warm run {0,1}
// (overlapping previous position 0), giving the request [2,12).
func TestOrderedAllocator_FragmentedIncidentGeometry(t *testing.T) {
	bound := []orderedAllocation{
		{id: "llama", start: 0, size: 2, positions: []int{0, 2}, evictable: true, boundAt: 100},
	}
	a := &orderedAllocator{chainSize: 12, bound: bound}
	plan := a.schedule(orderedRequest{size: 10}, 1)
	if plan.pending {
		t.Fatalf("unexpected pending: %s", plan.pendingReason)
	}
	if len(plan.evictions) != 1 || plan.evictions[0].id != "llama" {
		t.Fatalf("evictions = %+v, want exactly [llama]", plan.evictions)
	}
	if plan.placement != 2 {
		t.Errorf("placement = %d, want 2 (request takes [2,12) so llama keeps its warm node 0)",
			plan.placement)
	}
	if got := plan.evictions[0].newStart; got != 0 {
		t.Errorf("llama newStart = %d, want 0 (warm run {0,1})", got)
	}
	// Re-placing a fragmented allocation at a run beginning on its old
	// start position is a REAL eviction, not a no-op: {0,2} → {0,1} moves
	// the pod at position 2. The eviction must be reported (and counted
	// against the budget) even though newStart == old start.
	checkPlanInvariants(t, 12, bound, orderedRequest{size: 10}, plan)
}

func TestComputeFreeIntervals(t *testing.T) {
	cases := []struct {
		name      string
		chainSize int
		bound     []orderedAllocation
		want      []interval
	}{
		{"empty chain", 12, nil, []interval{{0, 12}}},
		{"single full", 12, []orderedAllocation{alloc("A", 0, 12)}, nil},
		{"left half", 12, []orderedAllocation{alloc("A", 0, 6)}, []interval{{6, 12}}},
		{"right half", 12, []orderedAllocation{alloc("A", 6, 6)}, []interval{{0, 6}}},
		{"middle gap", 12, []orderedAllocation{
			alloc("A", 0, 4), alloc("B", 8, 4),
		}, []interval{{4, 8}}},
		{"checkerboard 4", 4, []orderedAllocation{
			alloc("A", 0, 1), alloc("B", 2, 1),
		}, []interval{{1, 2}, {3, 4}}},
		{"unsorted input", 12, []orderedAllocation{
			alloc("B", 8, 4), alloc("A", 0, 4),
		}, []interval{{4, 8}}},
		{"adjacent merge boundaries", 12, []orderedAllocation{
			alloc("A", 0, 4), alloc("B", 4, 4),
		}, []interval{{8, 12}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := computeFreeIntervals(c.bound, c.chainSize)
			if !slices.EqualFunc(got, c.want, func(a, b interval) bool {
				return a.start == b.start && a.end == b.end
			}) {
				t.Errorf("got %+v, want %+v", got, c.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Section 12: First-fit-leftmost unit tests.
// ---------------------------------------------------------------------------

func TestFirstFitLeftmost(t *testing.T) {
	cases := []struct {
		name      string
		chainSize int
		bound     []orderedAllocation
		size      int
		wantStart int
		wantOk    bool
	}{
		{"empty chain size=8", 12, nil, 8, 0, true},
		{"empty chain size=12", 12, nil, 12, 0, true},
		{"empty chain size=13", 12, nil, 13, 0, false},
		{"empty chain size=0", 12, nil, 0, 0, false},
		{"left blocked", 12, []orderedAllocation{alloc("A", 0, 4)}, 4, 4, true},
		{"middle blocked", 12, []orderedAllocation{alloc("A", 4, 2)}, 4, 0, true},
		{"middle blocked, larger req", 12, []orderedAllocation{alloc("A", 4, 2)}, 6, 6, true},
		{"middle blocked, too large", 12, []orderedAllocation{alloc("A", 4, 2)}, 8, 0, false},
		{"checkerboard, no fit", 12, checkerboard(12), 2, 0, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			start, ok := firstFitLeftmost(c.bound, c.chainSize, c.size)
			if ok != c.wantOk {
				t.Errorf("ok = %v, want %v", ok, c.wantOk)
			}
			if ok && start != c.wantStart {
				t.Errorf("start = %d, want %d", start, c.wantStart)
			}
		})
	}
}
