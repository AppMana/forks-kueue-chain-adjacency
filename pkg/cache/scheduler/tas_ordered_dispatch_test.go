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

// End-to-end tests for the ordered-level dispatch in findTopologyAssignment.
// Drives the full path through TASFlavorCache → snapshot →
// FindTopologyAssignmentsForFlavor with an ordered chain-index level
// configured. Verifies the chain-adjacency invariant: rank N lands on
// chain-index start+N, and requests that can't form a contiguous run
// stay pending.

package scheduler

import (
	"strconv"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	kueue "sigs.k8s.io/kueue/apis/kueue/v1beta2"
	tasindexer "sigs.k8s.io/kueue/pkg/controller/tas/indexer"
	"sigs.k8s.io/kueue/pkg/resources"
	utiltesting "sigs.k8s.io/kueue/pkg/util/testing"
	testingnode "sigs.k8s.io/kueue/pkg/util/testingjobs/node"
	testingpod "sigs.k8s.io/kueue/pkg/util/testingjobs/pod"
)

const (
	tasOrderedChainName  = "topology.example.com/chain-name"
	tasOrderedChainIndex = "topology.example.com/chain-index"
)

// orderedChainLevels is the canonical 3-level layout used in these tests:
// chain-name (unordered) → chain-index (ordered) → hostname (unordered).
var orderedChainLevels = []string{
	tasOrderedChainName,
	tasOrderedChainIndex,
	corev1.LabelHostname,
}

// makeOrderedChainNode builds one node on the chain at the given chain-index,
// with the supplied per-node CPU.  Hostname is "h<idx>".
func makeOrderedChainNode(chainName string, idx int, cpu string) *corev1.Node {
	hostname := "h" + strconv.Itoa(idx)
	return testingnode.MakeNode(hostname).
		Label(tasOrderedChainName, chainName).
		Label(tasOrderedChainIndex, strconv.Itoa(idx)).
		Label(corev1.LabelHostname, hostname).
		StatusAllocatable(corev1.ResourceList{
			corev1.ResourceCPU:  resource.MustParse(cpu),
			corev1.ResourcePods: resource.MustParse("10"),
		}).
		Ready().
		Obj()
}

// occupyOrderedChainHost returns a non-TAS Pod that consumes the entire CPU
// of the named host, simulating an admitted workload on that chain-index
// without needing the full Workload/Admission machinery. Pod name is
// derived from the host name to avoid collisions.
func occupyOrderedChainHost(hostname string, cpu string) *corev1.Pod {
	return testingpod.MakePod("occupant-"+hostname, "default").
		NodeName(hostname).
		Request(corev1.ResourceCPU, cpu).
		StatusPhase(corev1.PodRunning).
		Obj()
}

// TestOrderedDispatch_FindTopologyAssignment exercises the ordered dispatch
// in findTopologyAssignment via FindTopologyAssignmentsForFlavor. Cases
// cover empty chain, occupied left half, fragmented chain (must reject),
// and a regression guard with Ordered=false.
func TestOrderedDispatch_FindTopologyAssignment(t *testing.T) {
	const chainName = "primary"
	type podSpec struct {
		hostname string
		cpu      string
	}
	cases := map[string]struct {
		ordered          []bool   // per-level Ordered flags; nil = all false
		occupiedHosts    []podSpec
		requestCount     int32
		wantHostnames    []string // expected ordered list of hostnames in the assignment; nil if want failure
		wantFailureMatch string   // substring expected in failure reason; "" = no check
	}{
		"empty chain, request size 6 → assigned h0..h5 in order": {
			ordered:       []bool{false, true, false},
			requestCount:  6,
			wantHostnames: []string{"h0", "h1", "h2", "h3", "h4", "h5"},
		},
		"occupied 0..2, request size 3 → assigned h3..h5": {
			ordered: []bool{false, true, false},
			occupiedHosts: []podSpec{
				{hostname: "h0", cpu: "1"},
				{hostname: "h1", cpu: "1"},
				{hostname: "h2", cpu: "1"},
			},
			requestCount:  3,
			wantHostnames: []string{"h3", "h4", "h5"},
		},
		"index 2 occupied, request size 5 → fails (no contiguous 5-run)": {
			ordered: []bool{false, true, false},
			occupiedHosts: []podSpec{
				{hostname: "h2", cpu: "1"},
			},
			requestCount:     5,
			wantFailureMatch: "no contiguous run",
		},
		"index 3 occupied, request size 3 → assigned h0..h2 (leftmost run)": {
			ordered: []bool{false, true, false},
			occupiedHosts: []podSpec{
				{hostname: "h3", cpu: "1"},
			},
			requestCount:  3,
			wantHostnames: []string{"h0", "h1", "h2"},
		},
		"unordered (default) preserves upstream behaviour": {
			ordered: nil, // no setLevelOrdered
			occupiedHosts: []podSpec{
				{hostname: "h2", cpu: "1"},
			},
			requestCount: 5,
			// Without ordered constraint, upstream picks 5 hosts with capacity in
			// lex order: h0, h1, h3, h4, h5. The chain-adjacency invariant is
			// violated but that's the upstream behaviour we shouldn't disturb
			// when Ordered=false.
			wantHostnames: []string{"h0", "h1", "h3", "h4", "h5"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, log := utiltesting.ContextWithLog(t)

			nodes := make([]*corev1.Node, 6)
			for i := 0; i < 6; i++ {
				nodes[i] = makeOrderedChainNode(chainName, i, "1")
			}

			pods := make([]*corev1.Pod, 0, len(tc.occupiedHosts))
			for _, occ := range tc.occupiedHosts {
				pods = append(pods, occupyOrderedChainHost(occ.hostname, occ.cpu))
			}

			initialObjects := make([]client.Object, 0, len(nodes)+len(pods))
			for i := range nodes {
				initialObjects = append(initialObjects, nodes[i])
			}
			for i := range pods {
				initialObjects = append(initialObjects, pods[i])
			}
			clientBuilder := utiltesting.NewClientBuilder()
			clientBuilder.WithObjects(initialObjects...)
			_ = tasindexer.SetupIndexes(ctx, utiltesting.AsIndexer(clientBuilder))
			c := clientBuilder.Build()

			tasCache := NewTASCache(c)
			for i := range nodes {
				tasCache.SyncNode(nodes[i])
			}
			for i := range pods {
				tasCache.Update(pods[i], log)
			}

			topo := topologyInformation{
				Levels:  orderedChainLevels,
				Ordered: tc.ordered,
			}
			flavor := flavorInformation{TopologyName: "ordered-test"}
			tasFlavorCache := tasCache.NewTASFlavorCache(topo, flavor)
			snapshot := tasFlavorCache.snapshot(log,
				tasCache.nodesCache.find(tasFlavorCache.flavor.NodeLabels, tasFlavorCache.topology.Levels))

			req := []TASPodSetRequests{{
				PodSet: &kueue.PodSet{
					Name: kueue.PodSetReference("main"),
					TopologyRequest: &kueue.PodSetTopologyRequest{
						Required:                    ptr.To(tasOrderedChainName),
						PodSetSliceRequiredTopology: ptr.To(tasOrderedChainIndex),
						PodSetSliceSize:             ptr.To(int32(1)),
					},
					Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{}},
				},
				SinglePodRequests: resources.Requests{corev1.ResourceCPU: 1000},
				Count:             tc.requestCount,
			}}

			got := snapshot.FindTopologyAssignmentsForFlavor(req)
			ps, ok := got[kueue.PodSetReference("main")]
			if !ok {
				t.Fatalf("no result for main podset; got %v", got)
			}

			if tc.wantFailureMatch != "" {
				if ps.FailureReason == "" {
					t.Fatalf("expected failure containing %q, got success: %+v",
						tc.wantFailureMatch, ps.TopologyAssignment)
				}
				if !contains(ps.FailureReason, tc.wantFailureMatch) {
					t.Errorf("failure reason = %q, want substring %q",
						ps.FailureReason, tc.wantFailureMatch)
				}
				return
			}
			if ps.FailureReason != "" {
				t.Fatalf("unexpected failure: %s", ps.FailureReason)
			}
			if ps.TopologyAssignment == nil {
				t.Fatalf("nil assignment")
			}
			if len(ps.TopologyAssignment.Domains) != len(tc.wantHostnames) {
				t.Fatalf("domains = %d, want %d (got %+v)",
					len(ps.TopologyAssignment.Domains), len(tc.wantHostnames),
					ps.TopologyAssignment.Domains)
			}
			for i, want := range tc.wantHostnames {
				got := ps.TopologyAssignment.Domains[i].Values[len(ps.TopologyAssignment.Domains[i].Values)-1]
				if got != want {
					t.Errorf("domain[%d] = %q, want %q", i, got, want)
				}
			}
		})
	}
}

// TestOrderedDispatch_Compaction drives compaction end-to-end through
// FindTopologyAssignmentsForFlavor with WithCompactionBudget. Exercises
// the budget knob, evictability, and priority gating on a 12-node chain
// with two pre-bound LWS-style workloads.
func TestOrderedDispatch_Compaction(t *testing.T) {
	const chainName = "primary"
	const chainSize = 12

	type wantEviction struct {
		ref string
	}
	cases := map[string]struct {
		bound            []boundOrderedAllocation
		occupiedHosts    []string // hostnames whose nodes already host a non-TAS pod consuming all CPU
		requestCount     int32
		requestPriority  int32
		budget           int
		wantPending      bool
		wantFailureMatch string
		wantEvictedRefs  []string // sorted; nil if no evictions expected
		wantPlacement    []string // ordered hostnames for the request
	}{
		"budget=0, fragmented chain → pending (compaction disabled)": {
			bound: []boundOrderedAllocation{
				{ref: "ns/lws-A", start: 0, size: 3, evictable: true, boundAt: 100},
				{ref: "ns/lws-B", start: 6, size: 3, evictable: true, boundAt: 200},
			},
			occupiedHosts: []string{
				"h0", "h1", "h2", // lws-A
				"h6", "h7", "h8", // lws-B
			},
			requestCount:     6,
			budget:           0,
			wantPending:      true,
			wantFailureMatch: "compaction disabled",
		},
		"budget=1, fragmented chain → evict newer lws-B, admit at 3..8": {
			bound: []boundOrderedAllocation{
				{ref: "ns/lws-A", start: 0, size: 3, evictable: true, boundAt: 100},
				{ref: "ns/lws-B", start: 6, size: 3, evictable: true, boundAt: 200},
			},
			occupiedHosts: []string{
				"h0", "h1", "h2", "h6", "h7", "h8",
			},
			requestCount: 6,
			budget:       1,
			// Default victim comparator: equal priority → newer boundAt is
			// cheaper. lws-B (200) is newer than lws-A (100), so the
			// allocator evicts lws-B (older lws-A is protected).  After
			// eviction, lws-A still sits at 0..2; the request takes the
			// next 6 contiguous indices and lws-B relocates to 9..11.
			wantEvictedRefs: []string{"ns/lws-B"},
			wantPlacement:   []string{"h3", "h4", "h5", "h6", "h7", "h8"},
		},
		"budget=1, both bound non-evictable → pending": {
			bound: []boundOrderedAllocation{
				{ref: "ns/jobset-A", start: 0, size: 3, evictable: false, boundAt: 100},
				{ref: "ns/jobset-B", start: 6, size: 3, evictable: false, boundAt: 200},
			},
			occupiedHosts: []string{
				"h0", "h1", "h2", "h6", "h7", "h8",
			},
			requestCount: 6,
			budget:       1,
			wantPending:  true,
		},
		"budget=1, victims higher-priority than request → pending": {
			bound: []boundOrderedAllocation{
				{ref: "ns/lws-A", start: 0, size: 3, evictable: true, priority: 200, boundAt: 100},
				{ref: "ns/lws-B", start: 6, size: 3, evictable: true, priority: 200, boundAt: 200},
			},
			occupiedHosts: []string{
				"h0", "h1", "h2", "h6", "h7", "h8",
			},
			requestCount:    6,
			requestPriority: 100, // lower than victim priorities
			budget:          1,
			wantPending:     true,
		},
		"budget=1, request priority high enough to evict → succeeds": {
			bound: []boundOrderedAllocation{
				{ref: "ns/lws-A", start: 0, size: 3, evictable: true, priority: 100, boundAt: 100},
				{ref: "ns/lws-B", start: 6, size: 3, evictable: true, priority: 100, boundAt: 200},
			},
			occupiedHosts: []string{
				"h0", "h1", "h2", "h6", "h7", "h8",
			},
			requestCount:    6,
			requestPriority: 200,
			budget:          1,
			// Same outcome as the equal-priority case: newer victim wins.
			wantEvictedRefs: []string{"ns/lws-B"},
			wantPlacement:   []string{"h3", "h4", "h5", "h6", "h7", "h8"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, log := utiltesting.ContextWithLog(t)

			nodes := make([]*corev1.Node, chainSize)
			for i := 0; i < chainSize; i++ {
				nodes[i] = makeOrderedChainNode(chainName, i, "1")
			}
			pods := make([]*corev1.Pod, 0, len(tc.occupiedHosts))
			for _, h := range tc.occupiedHosts {
				pods = append(pods, occupyOrderedChainHost(h, "1"))
			}

			initialObjects := make([]client.Object, 0, len(nodes)+len(pods))
			for i := range nodes {
				initialObjects = append(initialObjects, nodes[i])
			}
			for i := range pods {
				initialObjects = append(initialObjects, pods[i])
			}
			clientBuilder := utiltesting.NewClientBuilder()
			clientBuilder.WithObjects(initialObjects...)
			_ = tasindexer.SetupIndexes(ctx, utiltesting.AsIndexer(clientBuilder))
			c := clientBuilder.Build()

			tasCache := NewTASCache(c)
			for i := range nodes {
				tasCache.SyncNode(nodes[i])
			}
			for i := range pods {
				tasCache.Update(pods[i], log)
			}
			topo := topologyInformation{
				Levels:  orderedChainLevels,
				Ordered: []bool{false, true, false},
			}
			flavor := flavorInformation{TopologyName: "ordered-test"}
			tasFlavorCache := tasCache.NewTASFlavorCache(topo, flavor)
			snapshot := tasFlavorCache.snapshot(log,
				tasCache.nodesCache.find(tasFlavorCache.flavor.NodeLabels, tasFlavorCache.topology.Levels))
			snapshot.setBoundOrderedAllocations(tc.bound)

			req := []TASPodSetRequests{{
				PodSet: &kueue.PodSet{
					Name: kueue.PodSetReference("main"),
					TopologyRequest: &kueue.PodSetTopologyRequest{
						Required:                    ptr.To(tasOrderedChainName),
						PodSetSliceRequiredTopology: ptr.To(tasOrderedChainIndex),
						PodSetSliceSize:             ptr.To(int32(1)),
					},
					Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{}},
				},
				SinglePodRequests: resources.Requests{corev1.ResourceCPU: 1000},
				Count:             tc.requestCount,
			}}

			got := snapshot.FindTopologyAssignmentsForFlavor(req,
				WithCompactionBudget(tc.budget),
				WithRequestPriority(tc.requestPriority),
			)
			ps, ok := got[kueue.PodSetReference("main")]
			if !ok {
				t.Fatalf("no result for main podset; got %v", got)
			}

			if tc.wantPending {
				if ps.FailureReason == "" {
					t.Fatalf("expected pending, got success: %+v", ps.TopologyAssignment)
				}
				if tc.wantFailureMatch != "" && !contains(ps.FailureReason, tc.wantFailureMatch) {
					t.Errorf("failure reason = %q, want substring %q",
						ps.FailureReason, tc.wantFailureMatch)
				}
				return
			}
			if ps.FailureReason != "" {
				t.Fatalf("unexpected failure: %s", ps.FailureReason)
			}
			if ps.TopologyAssignment == nil {
				t.Fatalf("nil assignment")
			}
			gotPlacement := make([]string, 0, len(ps.TopologyAssignment.Domains))
			for _, d := range ps.TopologyAssignment.Domains {
				gotPlacement = append(gotPlacement, d.Values[len(d.Values)-1])
			}
			if len(gotPlacement) != len(tc.wantPlacement) {
				t.Errorf("placement count = %d, want %d (got %v, want %v)",
					len(gotPlacement), len(tc.wantPlacement), gotPlacement, tc.wantPlacement)
			} else {
				for i := range gotPlacement {
					if gotPlacement[i] != tc.wantPlacement[i] {
						t.Errorf("placement[%d] = %q, want %q", i, gotPlacement[i], tc.wantPlacement[i])
					}
				}
			}
			gotEvictedRefs := make([]string, 0, len(ps.CompactionEvictions))
			for _, ref := range ps.CompactionEvictions {
				gotEvictedRefs = append(gotEvictedRefs, string(ref))
			}
			sortStrings(gotEvictedRefs)
			wantEvictedRefs := append([]string(nil), tc.wantEvictedRefs...)
			sortStrings(wantEvictedRefs)
			if !equalStringSlices(gotEvictedRefs, wantEvictedRefs) {
				t.Errorf("evicted refs = %v, want %v", gotEvictedRefs, wantEvictedRefs)
			}
		})
	}
}

func sortStrings(s []string) {
	if len(s) <= 1 {
		return
	}
	// simple insertion sort to avoid pulling in sort here
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestPickOrderedContiguousRun unit-tests the contiguous-pick helper
// without the surrounding scaffolding.
func TestPickOrderedContiguousRun(t *testing.T) {
	mk := func(sliceStates ...int32) []*domain {
		out := make([]*domain, len(sliceStates))
		for i, s := range sliceStates {
			out[i] = &domain{
				levelValues: []string{"primary", strconv.Itoa(i)},
				sliceState:  s,
				state:       s,
			}
		}
		return out
	}
	cases := map[string]struct {
		domains       []*domain
		need          int32
		leaderCount   int32
		sliceSize     int32
		wantStartIdx  int  // -1 if want nil
		wantPickedLen int
	}{
		"empty layout, fits at 0": {
			domains: mk(1, 1, 1, 1, 1, 1), need: 4, sliceSize: 1, wantStartIdx: 0, wantPickedLen: 4,
		},
		"left half occupied, fits at 3": {
			domains: mk(0, 0, 0, 1, 1, 1), need: 3, sliceSize: 1, wantStartIdx: 3, wantPickedLen: 3,
		},
		"middle gap, fits in second run": {
			domains: mk(1, 1, 0, 1, 1, 1, 1), need: 4, sliceSize: 1, wantStartIdx: 3, wantPickedLen: 4,
		},
		"no contiguous run of 5": {
			domains: mk(1, 1, 0, 1, 1, 1), need: 5, sliceSize: 1, wantStartIdx: -1,
		},
		"runs separated by single occupied cell": {
			domains: mk(1, 1, 1, 0, 1, 1, 1), need: 3, sliceSize: 1, wantStartIdx: 0, wantPickedLen: 3,
		},
		"leader is placed at start of run": {
			domains: mk(0, 1, 1, 1, 1), need: 3, leaderCount: 1, sliceSize: 1, wantStartIdx: 1, wantPickedLen: 3,
		},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			s := &TASFlavorSnapshot{}
			got := s.pickOrderedContiguousRun(c.domains, c.need, c.leaderCount, c.sliceSize)
			if c.wantStartIdx == -1 {
				if got != nil {
					t.Fatalf("expected nil, got %d picked", len(got))
				}
				return
			}
			if got == nil {
				t.Fatalf("expected pick, got nil")
			}
			if len(got) != c.wantPickedLen {
				t.Errorf("picked count = %d, want %d", len(got), c.wantPickedLen)
			}
			gotStart, _ := strconv.Atoi(got[0].levelValues[1])
			if gotStart != c.wantStartIdx {
				t.Errorf("start idx = %d, want %d", gotStart, c.wantStartIdx)
			}
			if c.leaderCount > 0 {
				// Leader bookkeeping: the first picked domain has leaderState=1.
				if got[0].leaderState != 1 {
					t.Errorf("first picked leaderState = %d, want 1", got[0].leaderState)
				}
				for i := 1; i < len(got); i++ {
					if got[i].leaderState != 0 {
						t.Errorf("non-leader domain[%d] leaderState = %d, want 0",
							i, got[i].leaderState)
					}
				}
			}
		})
	}
}

func contains(s, substr string) bool {
	return len(substr) > 0 && len(s) >= len(substr) && stringContains(s, substr)
}

func stringContains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
