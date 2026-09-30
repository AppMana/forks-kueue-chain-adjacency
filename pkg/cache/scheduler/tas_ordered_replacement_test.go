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
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	kueue "sigs.k8s.io/kueue/apis/kueue/v1beta2"
	tasindexer "sigs.k8s.io/kueue/pkg/controller/tas/indexer"
	"sigs.k8s.io/kueue/pkg/resources"
	utiltas "sigs.k8s.io/kueue/pkg/util/tas"
	utiltesting "sigs.k8s.io/kueue/pkg/util/testing"
	utiltestingapi "sigs.k8s.io/kueue/pkg/util/testing/v1beta2"
)

// Node replacement keeps every healthy pod where it is and finds one new node
// for the unhealthy one. On an ordered level each pod is a pipeline rank bound
// to its chain position, so no single replacement keeps the ranks adjacent:
// replacing must fail, and fail-fast eviction requeues the workload for a
// whole contiguous re-placement.
//
// Observed 2026-09-29 23:58Z: the four-stage Flux.2 job held chain indices
// 7..10 (023, 022, 021, 004). A DiskPressure taint on 004 marked it unhealthy
// and the replacement put the last stage on 002, index 11, across the gap.
func TestOrderedReplacement_NeverPlacesAcrossTheChain(t *testing.T) {
	ctx, log := utiltesting.ContextWithLog(t)
	const chainSize = 12

	objects := make([]client.Object, 0, chainSize)
	nodes := make([]*corev1.Node, chainSize)
	for i := range chainSize {
		nodes[i] = makeOrderedChainNode("primary", i, "1")
		objects = append(objects, nodes[i])
	}
	// The three healthy stages keep running on h7..h9.
	for _, h := range []string{"h7", "h8", "h9"} {
		objects = append(objects, occupyOrderedChainHost(h, "1"))
	}
	clientBuilder := utiltesting.NewClientBuilder()
	clientBuilder.WithObjects(objects...)
	_ = tasindexer.SetupIndexes(ctx, utiltesting.AsIndexer(clientBuilder))
	c := clientBuilder.Build()

	tasCache := NewTASCache(c)
	for _, n := range nodes {
		tasCache.SyncNode(n)
	}
	for _, o := range objects[chainSize:] {
		tasCache.Update(o.(*corev1.Pod), log)
	}
	tasFlavorCache := tasCache.NewTASFlavorCache(
		topologyInformation{Levels: orderedChainLevels, Ordered: []bool{false, true, false}},
		flavorInformation{TopologyName: "ordered-test"},
	)
	snapshot := tasFlavorCache.snapshot(log,
		tasCache.nodesCache.find(tasFlavorCache.flavor.NodeLabels, tasFlavorCache.topology.Levels), nil)

	admitted := &utiltas.TopologyAssignment{Levels: []string{corev1.LabelHostname}}
	for _, h := range []string{"h7", "h8", "h9", "h10"} {
		admitted.Domains = append(admitted.Domains, utiltas.TopologyDomainAssignment{Values: []string{h}, Count: 1})
	}
	wl := utiltestingapi.MakeWorkload("flux2", "spellsource").
		Admission(utiltestingapi.MakeAdmission("tb-chain", "trainer").
			PodSets(utiltestingapi.MakePodSetAssignment("trainer").
				Count(4).
				TopologyAssignment(utiltas.V1Beta2From(admitted)).
				Obj()).
			Obj()).
		UnhealthyNodes("h10").
		Obj()

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
		Count:             1,
	}}

	got := snapshot.FindTopologyAssignmentsForFlavor(req, WithWorkload(wl))["trainer"]
	if got.FailureReason == "" {
		var hosts []string
		if got.TopologyAssignment != nil {
			for _, d := range got.TopologyAssignment.Domains {
				hosts = append(hosts, d.Values[len(d.Values)-1])
			}
		}
		t.Fatalf("replacement admitted %v; want a failure so the workload is re-placed contiguously", hosts)
	}
	if !strings.Contains(got.FailureReason, "ordered") {
		t.Errorf("failure reason = %q, want it to name the ordered level", got.FailureReason)
	}
}
