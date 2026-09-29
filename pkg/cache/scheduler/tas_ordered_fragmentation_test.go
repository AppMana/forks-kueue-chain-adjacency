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

// The zero-disruption placement keeps the chain's free space contiguous:
// among the feasible windows it prefers the one leaving the longest free run,
// then the fewest free runs, then the workload's warm positions, then a window
// touching a chain end. A single-node workload returning to a warm
// node in the middle of the chain otherwise strands the free cells on both
// sides of it, and a two-node job waits although enough nodes are idle.

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

func TestOrderedAllocator_SingleNodeReturnsToChainEndNotWarmMiddle(t *testing.T) {
	cases := map[string]struct {
		runtimeOn025  bool
		wantPlacement int
		// wantTwoNode is where a following two-node request lands, or -1
		// when it must wait.
		wantTwoNode int
	}{
		// Every free cell is an isolated hole, so no window leaves the free
		// space more usable; the warm node wins and the two-node job waits
		// until 025 is freed.
		"Unity runtime still holds 025": {runtimeOn025: true, wantPlacement: 5, wantTwoNode: -1},
		"Unity runtime moved off 025":   {runtimeOn025: false, wantPlacement: 11, wantTwoNode: 5},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			bound := liveChain(c.runtimeOn025)
			t.Logf("before: %s", renderChain(12, bound, map[int]string{5: "qwen38"}))
			a := &orderedAllocator{chainSize: 12, bound: bound}
			// The Qwen3.8 server re-admitted by its rollout, warm on 009.
			plan := a.schedule(orderedRequest{size: 1, preferredPositions: []int{5}}, orderedBudgetNoCompact)
			if plan.pending {
				t.Fatalf("unexpected pending: %s", plan.pendingReason)
			}
			if plan.placement != c.wantPlacement {
				t.Errorf("qwen38 placement = %d, want %d", plan.placement, c.wantPlacement)
			}
			after := append(append([]orderedAllocation{}, bound...),
				orderedAllocation{id: "qwen38", start: plan.placement, size: 1})
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

func TestOrderedAllocator_WarmthStillBreaksFragmentationTies(t *testing.T) {
	// Free {0..3} and {8..11}, both four wide. A two-node workload warm on
	// {8,9} returns there: every corner leaves the same free shape.
	runSchedulerTestCase(t, schedulerTestCase{
		chainSize:     12,
		bound:         []orderedAllocation{allocPinned("B", 4, 4)},
		req:           orderedRequest{size: 2, preferredPositions: []int{8, 9}},
		budget:        orderedBudgetNoCompact,
		wantPlacement: 8,
	})
}

func TestPickOrderedContiguousRun_KeepsFreeSpaceContiguous(t *testing.T) {
	// sliceState 1 = free chain node. Live chain with 025 freed and 009
	// vacated by the Qwen3.8 rollout: free {2}, {5,6}, {11}.
	states := []int32{0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1}
	domains := make([]*domain, len(states))
	for i, st := range states {
		domains[i] = &domain{levelValues: []string{"primary", strconv.Itoa(i)}, sliceState: st, state: st}
	}
	s := &TASFlavorSnapshot{}
	got := s.pickOrderedContiguousRun(domains, 1, 0, 1, []int{5})
	if got == nil {
		t.Fatal("expected a pick")
	}
	if start, _ := strconv.Atoi(got[0].levelValues[1]); start != 11 {
		t.Errorf("start = %d, want 11 (chain end; 009 would split the only two-wide run)", start)
	}
}
