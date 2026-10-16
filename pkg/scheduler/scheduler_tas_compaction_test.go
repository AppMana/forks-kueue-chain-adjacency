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
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
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
	"sigs.k8s.io/kueue/pkg/workload"
)

// TestScheduleForTASCompactionEvictsOnce reproduces a 12-position ordered
// chain where a 4-slice workload's only possible run is blocked by one
// evictable single-slice workload. Compaction must evict that workload once
// and keep evicting nobody while the eviction is in flight: the scheduler
// requeues the compacting workload immediately and re-plans every cycle
// against a snapshot that still holds the victim until its pods stop. Once
// the victim releases its quota the 4-slice workload is admitted into the
// run, and the victim is readmitted outside it.
func TestScheduleForTASCompactionEvictsOnce(t *testing.T) {
	const (
		chainName  = "appmana.com/tb-chain-name"
		chainIndex = "appmana.com/tb-chain-index"
		evictable  = "kueue.x-k8s.io/tas-ordered-evictable"
		ns         = "default"
	)
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
			Resource(corev1.ResourceCPU, "12").
			Obj()).
		Preemption(kueue.ClusterQueuePreemption{
			WithinClusterQueue:            kueue.PreemptionPolicyLowerPriority,
			MaxEvictionsPerSchedulingPass: ptr.To[int32](1),
		}).
		Obj()
	lq := utiltestingapi.MakeLocalQueue("tb-chain", ns).ClusterQueue(cq.Name).Obj()

	host := func(idx int) string { return "chain-" + strconv.Itoa(idx) }
	nodes := make([]corev1.Node, 0, 12)
	for i := range 12 {
		nodes = append(nodes, *testingnode.MakeNode(host(i)).
			Label(chainName, "primary").
			Label(chainIndex, strconv.Itoa(i)).
			Label(corev1.LabelHostname, host(i)).
			StatusAllocatable(corev1.ResourceList{
				corev1.ResourceCPU:  resource.MustParse("1"),
				corev1.ResourcePods: resource.MustParse("10"),
			}).
			Ready().
			Obj())
	}

	podSet := func(count int32) *utiltestingapi.PodSetWrapper {
		return utiltestingapi.MakePodSet(kueue.DefaultPodSetName, int(count)).
			RequiredTopologyRequest(chainName).
			SliceRequiredTopologyRequest(chainIndex).
			SliceSizeTopologyRequest(1).
			Request(corev1.ResourceCPU, "1")
	}
	// occupant is a single-slice workload admitted on the given chain
	// position at the given time.
	occupant := func(name string, idx int, canEvict bool, at time.Time) kueue.Workload {
		w := utiltestingapi.MakeWorkload(name, ns).
			Queue(kueue.LocalQueueName(lq.Name)).
			Priority(1000).
			PodSets(*podSet(1).Obj()).
			ReserveQuotaAt(
				utiltestingapi.MakeAdmission(kueue.ClusterQueueReference(cq.Name)).
					PodSets(utiltestingapi.MakePodSetAssignment(kueue.DefaultPodSetName).
						Assignment(corev1.ResourceCPU, kueue.ResourceFlavorReference(rf.Name), "1").
						TopologyAssignment(utiltestingapi.MakeTopologyAssignment([]string{corev1.LabelHostname}).
							Domain(utiltestingapi.MakeTopologyDomainAssignment([]string{host(idx)}, 1).Obj()).
							Obj()).
						Obj()).
					Obj(),
				at,
			).
			AdmittedAt(true, at)
		if canEvict {
			w = w.Annotation(evictable, "true")
		}
		return *w.Obj()
	}

	// Chain positions 0-4, 9 and 10 hold workloads compaction may not move;
	// the evictee holds 5, leaving 6-8 and 11 free. The only 4-position run
	// is 5-8, so the 4-slice workload needs the evictee moved to 11.
	const evicteeName = "progress-eval"
	workloads := []kueue.Workload{
		occupant("trainer-0", 0, false, now.Add(-time.Hour)),
		occupant("trainer-1", 1, false, now.Add(-time.Hour)),
		occupant("trainer-2", 2, false, now.Add(-time.Hour)),
		occupant("trainer-3", 3, false, now.Add(-time.Hour)),
		occupant("trainer-4", 4, false, now.Add(-time.Hour)),
		occupant("eval-9", 9, false, now.Add(-time.Hour)),
		occupant("eval-10", 10, false, now.Add(-time.Hour)),
		occupant(evicteeName, 5, true, now.Add(-time.Minute)),
		*utiltestingapi.MakeWorkload("styleguide", ns).
			UID("styleguide-uid").
			Queue(kueue.LocalQueueName(lq.Name)).
			Priority(1000).
			PodSets(*podSet(4).Obj()).
			Obj(),
	}

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
	compactionEvictions := func() int {
		t.Helper()
		n := 0
		for _, e := range recorder.RecordedEvents {
			if e.Key.Name == evicteeName && e.Reason == "EvictedDueToTASCompaction" {
				n++
			}
		}
		return n
	}
	evictionCount := func() int32 {
		t.Helper()
		wl := get(evicteeName)
		if wl.Status.SchedulingStats == nil {
			return 0
		}
		var n int32
		for _, e := range wl.Status.SchedulingStats.Evictions {
			if e.Reason == kueue.WorkloadEvictedByTASCompaction {
				n += e.Count
			}
		}
		return n
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
		return out
	}
	cycle := func() {
		t.Helper()
		scheduler.schedule(ctx)
		wg.Wait()
	}

	// Cycle 1: compaction evicts the workload blocking the run.
	cycle()
	if got := compactionEvictions(); got != 1 {
		t.Fatalf("After the first cycle: got %d compaction evictions of %s, want 1", got, evicteeName)
	}
	if hosts := assignedHosts("styleguide"); hosts != nil {
		t.Fatalf("styleguide admitted before its victim left: %v", hosts)
	}

	// The scheduler cache has not yet observed the eviction (watch lag):
	// the snapshot still shows the victim bound without an Evicted
	// condition. Re-planning must not evict it again.
	for range 3 {
		cycle()
	}
	if got := compactionEvictions(); got != 1 {
		t.Errorf("While the eviction is unobserved: got %d compaction evictions of %s, want 1", got, evicteeName)
	}

	// The cache observes the Evicted condition, but the victim still holds
	// its quota until its pods stop.
	cqCache.AddOrUpdateWorkload(log, get(evicteeName))
	for range 3 {
		cycle()
	}
	if got := compactionEvictions(); got != 1 {
		t.Errorf("While the evicted workload drains: got %d compaction evictions of %s, want 1", got, evicteeName)
	}
	if got := evictionCount(); got != 1 {
		t.Errorf("Eviction count of %s in schedulingStats: got %d, want 1", evicteeName, got)
	}
	if hosts := assignedHosts("styleguide"); hosts != nil {
		t.Fatalf("styleguide admitted before its victim left: %v", hosts)
	}

	// The victim's pods stopped: the workload controller releases its quota
	// and requeues it, as it does in production.
	evictee := get(evicteeName)
	workload.UnsetQuotaReservationWithCondition(evictee, "Pending", "Evicted", fakeClock.Now())
	if err := cl.Status().Update(ctx, evictee); err != nil {
		t.Fatalf("Releasing quota of %s: %v", evicteeName, err)
	}
	if err := cqCache.DeleteWorkload(log, workload.Key(evictee)); err != nil {
		t.Fatalf("Removing %s from cache: %v", evicteeName, err)
	}
	if err := qManager.AddOrUpdateWorkload(log, evictee.DeepCopy()); err != nil {
		t.Fatalf("Requeueing %s: %v", evicteeName, err)
	}

	// The 4-slice workload goes first and takes the run 5-8.
	cycle()
	wantRun := []string{host(5), host(6), host(7), host(8)}
	if diff := cmp.Diff(wantRun, assignedHosts("styleguide"), cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
		t.Fatalf("styleguide placement (-want,+got):\n%s", diff)
	}
	if hosts := assignedHosts(evicteeName); hosts != nil {
		t.Fatalf("%s readmitted before styleguide: %v", evicteeName, hosts)
	}

	// The evictee is readmitted outside the run.
	cycle()
	if diff := cmp.Diff([]string{host(11)}, assignedHosts(evicteeName)); diff != "" {
		t.Errorf("%s placement (-want,+got):\n%s", evicteeName, diff)
	}
	if got := compactionEvictions(); got != 1 {
		t.Errorf("At the end: got %d compaction evictions of %s, want 1", got, evicteeName)
	}
}
