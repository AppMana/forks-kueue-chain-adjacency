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
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	kueue "sigs.k8s.io/kueue/apis/kueue/v1beta2"
	tasindexer "sigs.k8s.io/kueue/pkg/controller/tas/indexer"
	"sigs.k8s.io/kueue/pkg/resources"
	utiltesting "sigs.k8s.io/kueue/pkg/util/testing"
)

// The last two chain positions (h10, h11) are not on the fast fabric: their
// nodes carry kueue.x-k8s.io/tas-ordered-max-run-size=2, so they only join
// runs of at most two positions, unless a run is longer than every stretch of
// the chain without such nodes.
//
// Observed 2026-09-30: the four-stage Flux.2 job was placed on chain indices
// 8..11 (022, 021, 004, 002); 004 and 002 have no Thunderbolt link, and the
// pipeline hung twice with 004 as its last stage.
func TestOrderedDispatch_MaxRunSizeNodes(t *testing.T) {
	const chainSize = 12
	const maxRunLabel = "kueue.x-k8s.io/tas-ordered-max-run-size"

	cases := map[string]struct {
		occupiedHosts   []string
		bound           []boundOrderedAllocation
		requestCount    int32
		budget          int
		wantPlacement   []string
		wantEvictedRefs []string
		wantPending     bool
	}{
		"four stages never take the restricted tail; no fast window free → pending": {
			// 3..7 busy: free fast positions are 0..2 and 8..9 only.
			occupiedHosts: []string{"h3", "h4", "h5", "h6", "h7"},
			requestCount:  4,
			wantPending:   true,
		},
		"two stages may use the restricted tail": {
			occupiedHosts: []string{"h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"},
			requestCount:  2,
			wantPlacement: []string{"h10", "h11"},
		},
		"a run longer than every fast stretch spans the whole chain": {
			requestCount:  11,
			wantPlacement: []string{"h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"},
		},
		"compaction moves a two-stage job onto the restricted tail to free a fast window": {
			occupiedHosts: []string{"h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7"},
			bound: []boundOrderedAllocation{
				{ref: "ns/pinned", start: 0, size: 6, evictable: false, boundAt: 100},
				{ref: "ns/small", start: 6, size: 2, evictable: true, boundAt: 200},
			},
			requestCount:    4,
			budget:          1,
			wantPlacement:   []string{"h6", "h7", "h8", "h9"},
			wantEvictedRefs: []string{"ns/small"},
		},
		"compaction never moves a three-stage job onto the restricted tail": {
			occupiedHosts: []string{"h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7"},
			bound: []boundOrderedAllocation{
				{ref: "ns/pinned", start: 0, size: 5, evictable: false, boundAt: 100},
				{ref: "ns/three", start: 5, size: 3, evictable: true, boundAt: 200},
			},
			requestCount: 4,
			budget:       1,
			wantPending:  true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, log := utiltesting.ContextWithLog(t)

			objects := make([]client.Object, 0, chainSize+len(tc.occupiedHosts))
			nodes := make([]*corev1.Node, chainSize)
			for i := range chainSize {
				nodes[i] = makeOrderedChainNode("primary", i, "1")
				if i >= 10 {
					nodes[i].Labels[maxRunLabel] = "2"
				}
				objects = append(objects, nodes[i])
			}
			pods := make([]*corev1.Pod, 0, len(tc.occupiedHosts))
			for _, h := range tc.occupiedHosts {
				p := occupyOrderedChainHost(h, "1")
				pods = append(pods, p)
				objects = append(objects, p)
			}
			clientBuilder := utiltesting.NewClientBuilder()
			clientBuilder.WithObjects(objects...)
			_ = tasindexer.SetupIndexes(ctx, utiltesting.AsIndexer(clientBuilder))
			c := clientBuilder.Build()

			tasCache := NewTASCache(c)
			for _, n := range nodes {
				tasCache.SyncNode(n)
			}
			for _, p := range pods {
				tasCache.Update(p, log)
			}
			tasFlavorCache := tasCache.NewTASFlavorCache(
				topologyInformation{Levels: orderedChainLevels, Ordered: []bool{false, true, false}},
				flavorInformation{TopologyName: "ordered-test"},
			)
			snapshot := tasFlavorCache.snapshot(log,
				tasCache.nodesCache.find(tasFlavorCache.flavor.NodeLabels, tasFlavorCache.topology.Levels), nil)
			if tc.bound != nil {
				snapshot.setBoundOrderedAllocations(tc.bound)
			}

			req := []TASPodSetRequests{{
				PodSet: &kueue.PodSet{
					Name: "trainer",
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

			got := snapshot.FindTopologyAssignmentsForFlavor(req, WithCompactionBudget(tc.budget))["trainer"]
			var hosts []string
			if got.TopologyAssignment != nil {
				for _, d := range got.TopologyAssignment.Domains {
					hosts = append(hosts, d.Values[len(d.Values)-1])
				}
			}
			if tc.wantPending {
				if got.FailureReason == "" {
					t.Fatalf("admitted %v; want pending", hosts)
				}
				return
			}
			if got.FailureReason != "" {
				t.Fatalf("unexpected failure: %s", got.FailureReason)
			}
			if !equalStringSlices(hosts, tc.wantPlacement) {
				t.Errorf("placement = %v, want %v", hosts, tc.wantPlacement)
			}
			evicted := make([]string, 0, len(got.CompactionEvictions))
			for _, ref := range got.CompactionEvictions {
				evicted = append(evicted, string(ref))
			}
			sortStrings(evicted)
			if !equalStringSlices(evicted, tc.wantEvictedRefs) {
				t.Errorf("evicted = %v, want %v", evicted, tc.wantEvictedRefs)
			}
		})
	}
}
