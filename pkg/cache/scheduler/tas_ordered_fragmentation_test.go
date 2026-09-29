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

package scheduler

import (
	"strconv"
	"strings"
	"testing"
)

// Zero-disruption placement on an ordered level. A workload returns to its
// warm nodes when it can; otherwise, and among equally warm windows, it takes
// the window leaving the longest free run, then the fewest free runs, then a
// window touching a chain end, then the leftmost. A single-node workload with
// no warm nodes therefore goes into the tightest gap or to a chain end, never
// into the middle of the only run a multi-node job could use.

// liveChain is the Thunderbolt chain as observed on 2026-09-29: 018(0) 027(1)
// 019(2) 008(3) 020(4) 009(5) 025(6) 023(7) 022(8) 021(9) 004(10) 002(11).
func liveChain(runtimeOn025 bool) []orderedAllocation {
	bound := []orderedAllocation{
		{id: "curriculum-test", start: 0, size: 2, priority: 1000, boundAt: 1},
		{id: "ideogram4-phase-a", start: 3, size: 2, priority: 2000, boundAt: 1},
		{id: "flux2-phase-a", start: 7, size: 4, priority: 2000, boundAt: 1},
	}
	if runtimeOn025 {
		// A Unity runtime pod holds 025's GPU outside Kueue's model.
		bound = append(bound, orderedAllocation{id: "__pinned_6__", start: 6, size: 1})
	}
	return bound
}

func renderChain(chainSize int, bound []orderedAllocation, extra map[int]string) string {
	cells := make([]string, chainSize)
	for i := range cells {
		cells[i] = "."
	}
	for _, b := range bound {
		for _, p := range b.occupied() {
			cells[p] = strings.SplitN(strings.Trim(b.id, "_"), "-", 2)[0]
		}
	}
	for p, name := range extra {
		cells[p] = name
	}
	out := make([]string, chainSize)
	for i, c := range cells {
		out[i] = strconv.Itoa(i) + ":" + c
	}
	return strings.Join(out, " ")
}

func TestOrderedAllocator_ColdSingleNodeGoesToChainEndNotMiddle(t *testing.T) {
	cases := map[string]struct {
		runtimeOn025  bool
		wantPlacement int
		// wantTwoNode is where a following two-node request lands, or -1
		// when it must wait.
		wantTwoNode int
	}{
		// Every free cell is an isolated hole; the chain end wins.
		"Unity runtime still holds 025": {runtimeOn025: true, wantPlacement: 11, wantTwoNode: -1},
		// 009+025 is the only two-wide run; the single node stays out of it.
		"Unity runtime moved off 025": {runtimeOn025: false, wantPlacement: 11, wantTwoNode: 5},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			bound := liveChain(c.runtimeOn025)
			t.Logf("before: %s", renderChain(12, bound, nil))
			a := &orderedAllocator{chainSize: 12, bound: bound}
			// A single-node workload with no warm nodes, e.g. the Qwen3.8
			// server started fresh after being scaled to zero.
			plan := a.schedule(orderedRequest{size: 1}, orderedBudgetNoCompact)
			if plan.pending {
				t.Fatalf("unexpected pending: %s", plan.pendingReason)
			}
			if plan.placement != c.wantPlacement {
				t.Errorf("placement = %d, want %d", plan.placement, c.wantPlacement)
			}
			after := append(append([]orderedAllocation{}, bound...),
				orderedAllocation{id: "single", start: plan.placement, size: 1})
			t.Logf("after:  %s", renderChain(12, after, nil))

			next := (&orderedAllocator{chainSize: 12, bound: after}).schedule(orderedRequest{size: 2}, orderedBudgetNoCompact)
			if c.wantTwoNode < 0 {
				if !next.pending {
					t.Errorf("two-node request placed at %d, want pending", next.placement)
				}
				return
			}
			if next.pending || next.placement != c.wantTwoNode {
				t.Errorf("two-node request = %+v, want placement %d", next, c.wantTwoNode)
			}
		})
	}
}

func TestOrderedAllocator_WarmNodesWinOverContiguity(t *testing.T) {
	// Qwen3.8 warm on 009 returns there even though that splits the only
	// two-wide free run: warm weights are the first criterion.
	runSchedulerTestCase(t, schedulerTestCase{
		chainSize:     12,
		bound:         liveChain(false),
		req:           orderedRequest{size: 1, preferredPositions: []int{5}},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 5,
	})
}

func TestOrderedAllocator_BestFitKeepsTheLargestFreeRun(t *testing.T) {
	// Free runs {0..5} and {8..10}. Leftmost first-fit would split the
	// six-wide run; a single-node request belongs in the three-wide run.
	bound := []orderedAllocation{allocPinned("B", 6, 2), allocPinned("C", 11, 1)}
	runSchedulerTestCase(t, schedulerTestCase{
		chainSize:     12,
		bound:         bound,
		req:           orderedRequest{size: 1},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 8,
	})
	// A three-wide request fills the three-wide run exactly.
	runSchedulerTestCase(t, schedulerTestCase{
		chainSize:     12,
		bound:         bound,
		req:           orderedRequest{size: 3},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 8,
	})
}

func TestOrderedAllocator_FragmentationBreaksWarmthTies(t *testing.T) {
	// A three-wide request warm on {0,1}: windows starting at 0 and at 1 are
	// both two cells warm. Starting at 0 keeps the free space in one run.
	runSchedulerTestCase(t, schedulerTestCase{
		chainSize:     12,
		bound:         []orderedAllocation{allocPinned("B", 5, 7)},
		req:           orderedRequest{size: 3, preferredPositions: []int{0, 1}},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 0,
	})
	// Free {1..3} and {9..11}: every corner leaves a three-wide run and two
	// runs in all. The window at the chain end wins over the leftmost one.
	runSchedulerTestCase(t, schedulerTestCase{
		chainSize:     12,
		bound:         []orderedAllocation{allocPinned("A", 0, 1), allocPinned("B", 4, 5)},
		req:           orderedRequest{size: 1},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 11,
	})
	// A cold single node takes the tightest gap: free {0,1} and {4..11}.
	runSchedulerTestCase(t, schedulerTestCase{
		chainSize:     12,
		bound:         []orderedAllocation{allocPinned("B", 2, 2)},
		req:           orderedRequest{size: 1},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 0,
	})
}

func TestPickOrderedContiguousRun_KeepsFreeSpaceContiguous(t *testing.T) {
	// sliceState 1 = free chain node. Live chain with 025 freed and 009
	// vacated: free {2}, {5,6}, {11}.
	states := []int32{0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1}
	domains := func() []*domain {
		out := make([]*domain, len(states))
		for i, st := range states {
			out[i] = &domain{levelValues: []string{"primary", strconv.Itoa(i)}, sliceState: st, state: st}
		}
		return out
	}
	s := &TASFlavorSnapshot{}
	start := func(preferred []int) int {
		got := s.pickOrderedContiguousRun(domains(), 1, 0, 1, preferred)
		if got == nil {
			t.Fatal("expected a pick")
		}
		n, _ := strconv.Atoi(got[0].levelValues[1])
		return n
	}
	if got := start(nil); got != 11 {
		t.Errorf("cold single node start = %d, want 11 (chain end; 009 would split the only two-wide run)", got)
	}
	if got := start([]int{5}); got != 5 {
		t.Errorf("warm single node start = %d, want 5 (its warm node)", got)
	}
}
