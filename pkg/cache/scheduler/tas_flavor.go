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
	"slices"
	"strconv"
	"sync"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	kueue "sigs.k8s.io/kueue/apis/kueue/v1beta2"
	"sigs.k8s.io/kueue/pkg/features"
	"sigs.k8s.io/kueue/pkg/resources"
	utiltas "sigs.k8s.io/kueue/pkg/util/tas"
	"sigs.k8s.io/kueue/pkg/workload"
)

// usageOp indicates whether we should add or subtract the usage.
type usageOp int

const (
	// add usage to the cache
	add usageOp = iota
	// subtract usage from the cache
	subtract
)

func (u usageOp) asSignedOne() int {
	if u == add {
		return 1
	}
	return -1
}

type flavorInformation struct {
	// Name indicates the name of the topology specified in the
	// ResourceFlavor spec.topologyName field.
	TopologyName kueue.TopologyReference

	// nodeLabels is a map of nodeLabels defined in the ResourceFlavor object.
	NodeLabels map[string]string
	// tolerations represents the list of tolerations specified for the resource
	// flavor
	Tolerations []corev1.Toleration
}

type topologyInformation struct {
	// levels is a list of levels defined in the Topology object referenced
	// by the flavor corresponding to the cache.
	Levels []string
	// Ordered parallels Levels: Ordered[i] indicates whether the i-th level
	// has a meaningful 1-D integer order derived from the level's node-label
	// value. Ordered levels require contiguous-run allocations (rank N goes
	// to start+N at the level).
	Ordered []bool
}

// wlBoundMeta captures the per-workload bookkeeping that ordered-level
// compaction needs: priority, admission time, and evictability. These are
// derived from the Workload object at addUsage time and stored alongside
// wlUsage so the snapshot can construct boundOrderedAllocation entries
// without re-reading the Workload during scheduling.
type wlBoundMeta struct {
	priority  int32
	boundAt   int64
	evictable bool
}

type TASFlavorCache struct {
	sync.RWMutex

	client client.Client

	// topology represents the part of the Topology specification, e.g. the list
	// of topology levels, relevant for TAS-scheduling.
	topology topologyInformation

	// flavor represents the part of the ResourceFlavor specification, e.g. the
	// list of node labels and tolerations, relevant for TAS-scheduling.
	flavor flavorInformation

	// usage maintains the usage per topology domain
	usage map[utiltas.TopologyDomainID]resources.Requests

	// wlUsage tracks the usage coming from workloads, so that we can make the
	// usage removal indempotent - skip if it was not added.
	wlUsage map[workload.Reference][]workload.TopologyDomainRequests

	// wlMeta stores the per-workload metadata used by ordered-level
	// compaction: priority, admission time, evictability. Populated in
	// parallel with wlUsage; default zero values yield "non-evictable,
	// priority 0, admitted now" which is safe for non-ordered topologies.
	wlMeta map[workload.Reference]wlBoundMeta

	// wlUsageHistory retains the recorded usage of recently-unbound
	// workloads (bounded FIFO, orderedUsageHistoryLimit entries) so
	// re-admission can prefer their previous — cache-warm — chain
	// positions. wlHistoryOrder tracks insertion order for eviction.
	wlUsageHistory map[workload.Reference][]workload.TopologyDomainRequests
	wlHistoryOrder []workload.Reference

	// nonTasUsageCache maintains the usage coming from non-TAS pods,
	// e.g. static Pods or DaemonSet pods.
	nonTasUsageCache *nonTasUsageCache
}

func (t *tasCache) NewTASFlavorCache(topologyInfo topologyInformation,
	flavorInfo flavorInformation) *TASFlavorCache {
	return &TASFlavorCache{
		client:           t.client,
		topology:         topologyInfo,
		flavor:           flavorInfo,
		usage:            make(map[utiltas.TopologyDomainID]resources.Requests),
		wlUsage:          make(map[workload.Reference][]workload.TopologyDomainRequests),
		wlMeta:           make(map[workload.Reference]wlBoundMeta),
		wlUsageHistory:   make(map[workload.Reference][]workload.TopologyDomainRequests),
		nonTasUsageCache: t.nonTasUsageCache,
	}
}

func (c *TASFlavorCache) NodeLabels() map[string]string {
	return c.flavor.NodeLabels
}

func (c *TASFlavorCache) Topology() kueue.TopologyReference {
	return c.flavor.TopologyName
}

func (c *TASFlavorCache) TopologyLevels() []string {
	return c.topology.Levels
}

func (c *TASFlavorCache) snapshot(
	log logr.Logger, nodes []*nodeInfo, aggregatedDomainUsages map[utiltas.TopologyDomainID]resources.Requests,
) *TASFlavorSnapshot {
	c.RLock()
	defer c.RUnlock()

	infoKV := []any{
		"nodeLabels", c.flavor.NodeLabels,
		"levels", c.topology.Levels,
		"nodeCount", len(nodes),
	}
	if features.Enabled(features.TASHandleOverlappingFlavors) {
		infoKV = append(infoKV, "crossFlavorAggregation", aggregatedDomainUsages != nil)
	}
	log.V(3).Info("Constructing TAS snapshot", infoKV...)

	snapshot := newTASFlavorSnapshot(log, c.flavor.TopologyName, c.topology.Levels, c.flavor.Tolerations)
	snapshot.setLevelOrdered(c.topology.Ordered)
	nodeToDomain := make(map[string]utiltas.TopologyDomainID)
	for _, node := range nodes {
		nodeToDomain[node.Name] = snapshot.addNode(node)
	}
	snapshot.initialize()

	tasDomainUsages := c.usage
	if features.Enabled(features.TASHandleOverlappingFlavors) && aggregatedDomainUsages != nil {
		tasDomainUsages = aggregatedDomainUsages
	}
	for domainID, usage := range tasDomainUsages {
		snapshot.addTASUsage(domainID, usage)
	}
	c.nonTasUsageCache.forEachNodeUsage(func(nodeName string, usage resources.Requests) {
		if domainID, ok := nodeToDomain[nodeName]; ok {
			snapshot.addNonTASUsage(domainID, usage)
		}
	})
	// Build boundOrderedAllocations only when the topology has an ordered
	// level, since it's the sole consumer. For each bound workload, derive
	// (start, size) from its per-domain placement.
	if snapshot.hasOrderedLevels() {
		snapshot.setBoundOrderedAllocations(c.buildBoundOrderedAllocations(snapshot))
		snapshot.setLastKnownOrderedPositions(c.buildLastKnownOrderedPositions(snapshot))
	}
	return snapshot
}

// buildBoundOrderedAllocations turns the cache's wlUsage + wlMeta into the
// ordered-allocation shape the snapshot's compaction path consumes. A
// workload's chain-index positions are the integer values its per-domain
// placement carries at the ordered level — the EXACT positions, not their
// min..max span: a workload whose pods drifted to {0,2} occupies two
// cells (and needs two contiguous cells when re-placed), not three.
// Workloads whose placement isn't on the ordered level (e.g. they're
// admitted on a different flavor or before the snapshot's nodes existed)
// are dropped.
func (c *TASFlavorCache) buildBoundOrderedAllocations(s *TASFlavorSnapshot) []boundOrderedAllocation {
	orderedIdx := s.firstOrderedLevelIdx()
	if orderedIdx < 0 {
		return nil
	}
	if orderedIdx >= len(c.topology.Levels) {
		return nil
	}
	out := make([]boundOrderedAllocation, 0, len(c.wlUsage))
	for ref, requests := range c.wlUsage {
		indices := orderedIndicesForUsage(s, orderedIdx, requests)
		if len(indices) == 0 {
			continue
		}
		meta := c.wlMeta[ref]
		out = append(out, boundOrderedAllocation{
			ref:       ref,
			start:     indices[0],
			size:      len(indices),
			positions: indices,
			priority:  meta.priority,
			boundAt:   meta.boundAt,
			evictable: meta.evictable,
		})
	}
	return out
}

// orderedIndicesForUsage resolves the Ordered-level integer index values a
// set of per-domain requests occupies, deduplicated and sorted ascending.
func orderedIndicesForUsage(s *TASFlavorSnapshot, orderedIdx int, requests []workload.TopologyDomainRequests) []int {
	var indices []int
	seen := make(map[int]bool, len(requests))
	for _, req := range requests {
		// req.Values may be at the lowest level only when
		// isLowestLevelNode is true; resolve via leaf lookup so the
		// chain-index value is recoverable in either encoding.
		leaf, ok := s.leaves[utiltas.DomainID(req.Values)]
		if !ok || orderedIdx >= len(leaf.levelValues) {
			continue
		}
		n, err := strconv.Atoi(leaf.levelValues[orderedIdx])
		if err != nil || seen[n] {
			continue
		}
		seen[n] = true
		indices = append(indices, n)
	}
	slices.Sort(indices)
	return indices
}

// buildLastKnownOrderedPositions converts the usage history of
// recently-unbound workloads into Ordered-level index values, feeding the
// cache-warmth preference on re-admission. Workloads currently bound are
// skipped (their live positions are in boundOrderedAllocations).
func (c *TASFlavorCache) buildLastKnownOrderedPositions(s *TASFlavorSnapshot) map[workload.Reference][]int {
	orderedIdx := s.firstOrderedLevelIdx()
	if orderedIdx < 0 || len(c.wlUsageHistory) == 0 {
		return nil
	}
	out := make(map[workload.Reference][]int, len(c.wlUsageHistory))
	for ref, requests := range c.wlUsageHistory {
		if _, bound := c.wlUsage[ref]; bound {
			continue
		}
		if indices := orderedIndicesForUsage(s, orderedIdx, requests); len(indices) > 0 {
			out[ref] = indices
		}
	}
	return out
}

func (c *TASFlavorCache) addUsage(log logr.Logger, key workload.Reference, topologyRequests []workload.TopologyDomainRequests) {
	c.addUsageWithMeta(log, key, topologyRequests, wlBoundMeta{})
}

// addUsageWithMeta records bound usage along with the per-workload metadata
// that ordered-level compaction reads. Use this from production paths that
// have access to the Workload (priority, admission time, evictability);
// addUsage is a thin shim retained for callers that don't.
func (c *TASFlavorCache) addUsageWithMeta(
	log logr.Logger,
	key workload.Reference,
	topologyRequests []workload.TopologyDomainRequests,
	meta wlBoundMeta,
) {
	if _, found := c.wlUsage[key]; found {
		log.V(2).Info("Workload usage already exists in TAS flavor cache, self-healing by replacing it", "workload", key)
		c.removeUsage(log, key)
	}
	c.wlUsage[key] = slices.Clone(topologyRequests)
	if c.wlMeta == nil {
		c.wlMeta = make(map[workload.Reference]wlBoundMeta)
	}
	c.wlMeta[key] = meta
	c.updateUsage(topologyRequests, add)
}

func (c *TASFlavorCache) removeUsage(log logr.Logger, key workload.Reference) {
	value, found := c.wlUsage[key]
	if !found {
		log.V(2).Info("Workload usage not found during removal from TAS flavor cache", "workload", key)
		return
	}
	c.rememberUsage(key, value)
	c.updateUsage(value, subtract)
	delete(c.wlUsage, key)
	delete(c.wlMeta, key)
}

// orderedUsageHistoryLimit bounds wlUsageHistory. 256 covers far more
// unbind events than a chain can host between the eviction and the
// re-admission of the same workload.
const orderedUsageHistoryLimit = 256

// rememberUsage stashes a workload's recorded usage as it unbinds, so a
// later re-admission can prefer the same — cache-warm — chain positions.
// Bounded FIFO: the oldest remembered workload is dropped past the limit.
func (c *TASFlavorCache) rememberUsage(key workload.Reference, requests []workload.TopologyDomainRequests) {
	if c.wlUsageHistory == nil {
		c.wlUsageHistory = make(map[workload.Reference][]workload.TopologyDomainRequests)
	}
	if _, exists := c.wlUsageHistory[key]; !exists {
		c.wlHistoryOrder = append(c.wlHistoryOrder, key)
		if len(c.wlHistoryOrder) > orderedUsageHistoryLimit {
			oldest := c.wlHistoryOrder[0]
			c.wlHistoryOrder = c.wlHistoryOrder[1:]
			delete(c.wlUsageHistory, oldest)
		}
	}
	c.wlUsageHistory[key] = slices.Clone(requests)
}

func (c *TASFlavorCache) updateUsage(topologyRequests []workload.TopologyDomainRequests, op usageOp) {
	c.Lock()
	defer c.Unlock()
	for _, tr := range topologyRequests {
		domainID := utiltas.DomainID(tr.Values)
		_, found := c.usage[domainID]
		if !found {
			c.usage[domainID] = resources.Requests{}
		}
		if op == subtract {
			c.usage[domainID].Sub(tr.TotalRequests())
			c.usage[domainID].Sub(resources.Requests{corev1.ResourcePods: int64(tr.Count)})
		} else {
			c.usage[domainID].Add(tr.TotalRequests())
			c.usage[domainID].Add(resources.Requests{corev1.ResourcePods: int64(tr.Count)})
		}
	}
}

type nodeInfo struct {
	// Name holds the node's name, used to evaluate node affinity.
	Name string

	// Labels are used to match Topology levels and NodeSelectors.
	Labels map[string]string

	// Taints are used to check tolerations.
	Taints []corev1.Taint

	// Allocatable capacity from Status.Allocatable.
	Allocatable corev1.ResourceList
}

func newNodeInfo(node *corev1.Node) *nodeInfo {
	return &nodeInfo{
		Name:        node.Name,
		Labels:      node.Labels,
		Taints:      node.Spec.Taints,
		Allocatable: node.Status.Allocatable,
	}
}

func (ni *nodeInfo) toNode() *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   ni.Name,
			Labels: ni.Labels,
		},
		Spec: corev1.NodeSpec{
			Taints: ni.Taints,
		},
		Status: corev1.NodeStatus{
			Allocatable: ni.Allocatable,
		},
	}
}
