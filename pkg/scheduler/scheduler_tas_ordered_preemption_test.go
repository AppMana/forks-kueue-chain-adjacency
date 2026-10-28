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
	"context"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	kueue "sigs.k8s.io/kueue/apis/kueue/v1beta2"
	qcache "sigs.k8s.io/kueue/pkg/cache/queue"
	schdcache "sigs.k8s.io/kueue/pkg/cache/scheduler"
	tasindexer "sigs.k8s.io/kueue/pkg/controller/tas/indexer"
	preemptexpectations "sigs.k8s.io/kueue/pkg/scheduler/preemption/expectations"
	"sigs.k8s.io/kueue/pkg/util/routine"
	utiltas "sigs.k8s.io/kueue/pkg/util/tas"
	utiltesting "sigs.k8s.io/kueue/pkg/util/testing"
	utiltestingapi "sigs.k8s.io/kueue/pkg/util/testing/v1beta2"
	testingnode "sigs.k8s.io/kueue/pkg/util/testingjobs/node"
	testingpod "sigs.k8s.io/kueue/pkg/util/testingjobs/pod"
	"sigs.k8s.io/kueue/pkg/workload"
)

// TestScheduleForTASOrderedPriorityPreemption models tb-chain on
// appmana-cluster-03: twelve chain positions, the last two on hosts without
// a Thunderbolt link (tas-ordered-max-run-size=2) that a multi-node trainer
// excludes by node affinity. Priority-2000 trainers hold most of the chain
// and must never move; priority-1500 single-pod evaluations hold the rest.
// A pending two-slice trainer is short of pods quota, so it must preempt
// exactly the evaluations that open a contiguous two-position run.
func TestScheduleForTASOrderedPriorityPreemption(t *testing.T) {
	const (
		chainName  = "appmana.com/tb-chain-name"
		chainIndex = "appmana.com/tb-chain-index"
		maxRunSize = "kueue.x-k8s.io/tas-ordered-max-run-size"
		evictable  = "kueue.x-k8s.io/tas-ordered-evictable"
		gpu        = corev1.ResourceName("nvidia.com/gpu")
		ns         = "spellsource"
		pending    = "krea2-edit-sil-wb-s8-c512"
		queuedEval = "progress-eval-qwen21-s8-c512-cv2-ta-written"
	)
	// Chain index -> host, as labelled on appmana-cluster-03.
	hostAt := map[int]string{
		0: "appmana-018", 1: "appmana-027", 2: "appmana-019", 3: "appmana-008",
		4: "appmana-020", 5: "appmana-009", 6: "appmana-025", 7: "appmana-023",
		8: "appmana-022", 9: "appmana-021", 10: "appmana-004", 11: "appmana-002",
	}
	type occupant struct {
		name string
		idxs []int
	}
	// The trainers running on 2026-10-28.
	trainers := []occupant{
		{"krea2-s8-c512-cv2-written", []int{0, 1}},
		{"iris-s16-c1024", []int{2}},
		{"qwen21-s8-c512-cv2-ta-written", []int{5}},
		{"iris-s16-c1024-fft", []int{7, 8, 9}},
	}
	cases := map[string]struct {
		evals []occupant
		// nonKueueCPU is the CPU requested by pods Kueue does not manage,
		// per chain index.
		nonKueueCPU map[int]string
		wantEvicted []string
		wantHosts   []string
	}{
		"a free position next to an evaluation": {
			evals: []occupant{
				{"ref-t2i-edit-trials", []int{3}},
				{"final-eval-fibo2-edit-sil-fuschia", []int{6}},
				{"final-eval-pixeldit", []int{10}},
				{"final-eval-fibo2-edit-sil-wb", []int{11}},
			},
			wantEvicted: []string{"ref-t2i-edit-trials"},
			wantHosts:   []string{hostAt[3], hostAt[4]},
		},
		"two adjacent evaluations": {
			evals: []occupant{
				{"ref-t2i-edit-trials", []int{3}},
				{"progress-eval-iris", []int{4}},
				{"final-eval-fibo2-edit-sil-fuschia", []int{6}},
				{"final-eval-pixeldit", []int{10}},
				{"final-eval-fibo2-edit-sil-wb", []int{11}},
			},
			wantEvicted: []string{"progress-eval-iris", "ref-t2i-edit-trials"},
			wantHosts:   []string{hostAt[3], hostAt[4]},
		},
		// The free position's CPU is taken by pods outside Kueue, so no
		// eviction of evaluations can open a run: none is issued.
		"the free position is full of pods outside Kueue": {
			evals: []occupant{
				{"ref-t2i-edit-trials", []int{3}},
				{"final-eval-fibo2-edit-sil-fuschia", []int{6}},
				{"final-eval-pixeldit", []int{10}},
				{"final-eval-fibo2-edit-sil-wb", []int{11}},
			},
			nonKueueCPU: map[int]string{4: "25"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			now := time.Now().Truncate(time.Second)
			topology := utiltestingapi.MakeTopology("tb-chain").
				Levels(chainName, chainIndex, corev1.LabelHostname).
				Obj()
			topology.Spec.Levels[1].Ordered = true
			rf := utiltestingapi.MakeResourceFlavor("tb-chain-primary").
				NodeLabel(chainName, "primary").
				TopologyName(topology.Name).
				Obj()
			cq := utiltestingapi.MakeClusterQueue("tb-chain").
				ResourceGroup(*utiltestingapi.MakeFlavorQuotas(rf.Name).
					Resource(corev1.ResourceCPU, "768").
					Resource(corev1.ResourcePods, "12").
					Resource(gpu, "288").
					Obj()).
				Preemption(kueue.ClusterQueuePreemption{
					WithinClusterQueue:            kueue.PreemptionPolicyLowerPriority,
					ReclaimWithinCohort:           kueue.PreemptionPolicyNever,
					MaxEvictionsPerSchedulingPass: ptr.To[int32](1),
				}).
				Obj()
			lq := utiltestingapi.MakeLocalQueue("tb-chain", ns).ClusterQueue(cq.Name).Obj()

			nodes := make([]corev1.Node, 0, len(hostAt))
			for idx := range 12 {
				n := testingnode.MakeNode(hostAt[idx]).
					Label(chainName, "primary").
					Label(chainIndex, strconv.Itoa(idx)).
					Label(corev1.LabelHostname, hostAt[idx]).
					StatusAllocatable(corev1.ResourceList{
						corev1.ResourceCPU:  resource.MustParse("32"),
						corev1.ResourcePods: resource.MustParse("110"),
						gpu:                 resource.MustParse("24"),
					}).
					Ready()
				if idx >= 10 {
					n = n.Label(maxRunSize, "2")
				}
				nodes = append(nodes, *n.Obj())
			}

			podSet := func(count int32) *utiltestingapi.PodSetWrapper {
				return utiltestingapi.MakePodSet("main", int(count)).
					RequiredTopologyRequest(chainName).
					SliceRequiredTopologyRequest(chainIndex).
					SliceSizeTopologyRequest(1).
					Request(corev1.ResourceCPU, "8").
					Request(gpu, "24")
			}
			admitted := func(o occupant, priority int32, canEvict bool, at time.Time) kueue.Workload {
				ta := utiltestingapi.MakeTopologyAssignment([]string{corev1.LabelHostname})
				for _, idx := range o.idxs {
					ta = ta.Domain(utiltestingapi.MakeTopologyDomainAssignment([]string{hostAt[idx]}, 1).Obj())
				}
				count := int32(len(o.idxs))
				ps := podSet(count)
				if canEvict {
					ps = ps.Annotations(map[string]string{evictable: "true"})
				}
				return *utiltestingapi.MakeWorkload(o.name, ns).
					Queue(kueue.LocalQueueName(lq.Name)).
					Priority(priority).
					PodSets(*ps.Obj()).
					ReserveQuotaAt(
						utiltestingapi.MakeAdmission(kueue.ClusterQueueReference(cq.Name)).
							PodSets(utiltestingapi.MakePodSetAssignment("main").
								Assignment(corev1.ResourceCPU, kueue.ResourceFlavorReference(rf.Name), strconv.Itoa(8*int(count))).
								Assignment(gpu, kueue.ResourceFlavorReference(rf.Name), strconv.Itoa(24*int(count))).
								Assignment(corev1.ResourcePods, kueue.ResourceFlavorReference(rf.Name), strconv.Itoa(int(count))).
								Count(count).
								TopologyAssignment(ta.Obj()).
								Obj()).
							Obj(),
						at,
					).
					AdmittedAt(true, at).
					Obj()
			}

			var workloads []kueue.Workload
			var running []string
			for _, o := range trainers {
				workloads = append(workloads, admitted(o, 2000, false, now.Add(-48*time.Hour)))
				running = append(running, o.name)
			}
			for i, o := range tc.evals {
				workloads = append(workloads, admitted(o, 1500, true, now.Add(-time.Duration(60+i)*time.Minute)))
				running = append(running, o.name)
			}
			workloads = append(workloads, *utiltestingapi.MakeWorkload(pending, ns).
				UID("pending-uid").
				Queue(kueue.LocalQueueName(lq.Name)).
				Priority(2000).
				PodSets(*podSet(2).
					SubGroupCount(ptr.To[int32](2)).
					SubGroupIndexLabel(ptr.To("jobset.sigs.k8s.io/job-index")).
					PodIndexLabel(ptr.To("batch.kubernetes.io/job-completion-index")).
					RequiredNodeSelectorRequirement(maxRunSize, corev1.NodeSelectorOpDoesNotExist).
					Obj()).
				Obj())
			// An evaluation queued behind the trainer must not take the
			// positions the trainer's preemptions free.
			workloads = append(workloads, *utiltestingapi.MakeWorkload(queuedEval, ns).
				UID("queued-eval-uid").
				Queue(kueue.LocalQueueName(lq.Name)).
				Priority(1500).
				Creation(now).
				PodSets(*podSet(1).Annotations(map[string]string{evictable: "true"}).Obj()).
				Obj())

			ctx, log := utiltesting.ContextWithLog(t)
			clientBuilder := utiltesting.NewClientBuilder().
				WithLists(
					&kueue.WorkloadList{Items: workloads},
					&corev1.NodeList{Items: nodes},
				).
				WithObjects(utiltesting.MakeNamespace(ns), topology, rf, cq, lq).
				WithInterceptorFuncs(interceptor.Funcs{SubResourcePatch: utiltesting.TreatSSAAsStrategicMerge}).
				WithStatusSubresource(&kueue.Workload{})
			_ = tasindexer.SetupIndexes(ctx, utiltesting.AsIndexer(clientBuilder))
			cl := clientBuilder.Build()
			recorder := &utiltesting.EventRecorder{}
			fakeClock := testingclock.NewFakeClock(now)
			cqCache := schdcache.New(cl)
			expectations := preemptexpectations.New()
			qManager := qcache.NewManagerForUnitTests(cl, cqCache, qcache.WithClock(fakeClock), qcache.WithPreemptionExpectations(expectations))
			for i := range nodes {
				cqCache.TASCache().SyncNode(&nodes[i])
			}
			for idx, cpu := range tc.nonKueueCPU {
				pod := testingpod.MakePod("outside-kueue-"+strconv.Itoa(idx), "default").
					NodeName(hostAt[idx]).
					Request(corev1.ResourceCPU, cpu).
					StatusPhase(corev1.PodRunning).
					Obj()
				cqCache.TASCache().Update(pod, log)
			}
			cqCache.AddOrUpdateResourceFlavor(log, rf.DeepCopy())
			cqCache.AddOrUpdateTopology(log, topology.DeepCopy())
			if err := cqCache.AddClusterQueue(ctx, cq.DeepCopy()); err != nil {
				t.Fatalf("Inserting clusterQueue in cache: %v", err)
			}
			if err := qManager.AddClusterQueue(ctx, cq.DeepCopy()); err != nil {
				t.Fatalf("Inserting clusterQueue in manager: %v", err)
			}
			for i := range workloads {
				if workload.HasQuotaReservation(&workloads[i]) {
					cqCache.AddOrUpdateWorkload(log, workloads[i].DeepCopy())
				}
			}
			if err := qManager.AddLocalQueue(ctx, lq.DeepCopy()); err != nil {
				t.Fatalf("Inserting localQueue in manager: %v", err)
			}

			scheduler := New(qManager, cqCache, cl, recorder, WithClock(t, fakeClock), WithPreemptionExpectations(expectations))
			wg := sync.WaitGroup{}
			scheduler.setAdmissionRoutineWrapper(routine.NewWrapper(
				func() { wg.Add(1) },
				func() { wg.Done() },
			))
			ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
			go qManager.CleanUpOnContext(ctx)

			get := func(name string) *kueue.Workload {
				t.Helper()
				var wl kueue.Workload
				if err := cl.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, &wl); err != nil {
					t.Fatalf("Getting workload %s: %v", name, err)
				}
				return &wl
			}
			evicted := func() []string {
				t.Helper()
				var out []string
				for _, w := range running {
					if workload.IsEvicted(get(w)) {
						out = append(out, w)
					}
				}
				slices.Sort(out)
				return out
			}
			assignedHosts := func(name string) []string {
				t.Helper()
				wl := get(name)
				if wl.Status.Admission == nil {
					return nil
				}
				var out []string
				for _, psa := range wl.Status.Admission.PodSetAssignments {
					ta := utiltas.InternalFrom(psa.TopologyAssignment)
					if ta == nil {
						continue
					}
					for _, d := range ta.Domains {
						out = append(out, d.Values[len(d.Values)-1])
					}
				}
				slices.Sort(out)
				return out
			}
			cycle := func() {
				t.Helper()
				scheduler.schedule(ctx)
				wg.Wait()
			}

			cycle()
			if got := evicted(); !slices.Equal(got, tc.wantEvicted) {
				t.Fatalf("After the first cycle: evicted %v, want %v", got, tc.wantEvicted)
			}
			if len(tc.wantEvicted) == 0 {
				// Nothing was evicted, so nothing requeues the pending
				// workload; another cycle would block on an empty queue.
				if got := assignedHosts(pending); got != nil {
					t.Errorf("%s admitted on %v, want pending", pending, got)
				}
				return
			}

			// While the victims drain, the trainer keeps its claim.
			for range 3 {
				cycle()
			}
			if got := evicted(); !slices.Equal(got, tc.wantEvicted) {
				t.Fatalf("While the victims drain: evicted %v, want %v", got, tc.wantEvicted)
			}
			if got := assignedHosts(queuedEval); got != nil {
				t.Fatalf("%s admitted on %v while the trainer's victims drain", queuedEval, got)
			}

			// The victims' pods stop: the workload controller releases
			// their quota and requeues them.
			for _, victim := range tc.wantEvicted {
				evictee := get(victim)
				workload.UnsetQuotaReservationWithCondition(evictee, "Pending", "Evicted", fakeClock.Now())
				if err := cl.Status().Update(ctx, evictee); err != nil {
					t.Fatalf("Releasing quota of %s: %v", victim, err)
				}
				if err := cqCache.DeleteWorkload(log, workload.Key(evictee)); err != nil {
					t.Fatalf("Removing %s from cache: %v", victim, err)
				}
				if err := qManager.AddOrUpdateWorkload(log, evictee.DeepCopy()); err != nil {
					t.Fatalf("Requeueing %s: %v", victim, err)
				}
			}

			cycle()
			if got := assignedHosts(pending); !slices.Equal(got, tc.wantHosts) {
				t.Errorf("%s placement: got %v, want %v", pending, got, tc.wantHosts)
			}
			if got := evicted(); !slices.Equal(got, tc.wantEvicted) {
				t.Errorf("After the second cycle: evicted %v, want only %v", got, tc.wantEvicted)
			}
		})
	}
}
