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

// Tests for the ordered-level wiring inside TASFlavorSnapshot. Focuses on
// the chain-adjacency invariant: when a Topology level is marked Ordered,
// children at that level are sorted numerically (so chain-index "10" comes
// after "2", not before), and the final TopologyAssignment is emitted in
// that numeric order — which the rank-aware ungater reads as
// rank N → start+N.

package scheduler

import (
	"fmt"
	"strconv"
	"testing"

	corev1 "k8s.io/api/core/v1"

	utiltesting "sigs.k8s.io/kueue/pkg/util/testing"
	"sigs.k8s.io/kueue/pkg/util/testingjobs/node"
)

// chainNodes builds a 12-node Thunderbolt-style chain. Each node is on a
// single chain ("primary"), at chain-index 0..11, and on its own hostname.
// Returns nodes in REVERSED order to defeat any accidental insertion-order
// dependency in the snapshot — initialise() must produce numeric order
// regardless of insertion order.
func chainNodes(chainName string, count int) []corev1.Node {
	out := make([]corev1.Node, count)
	for i := count - 1; i >= 0; i-- {
		hostname := fmt.Sprintf("h%d", i)
		out[count-1-i] = *node.MakeNode(hostname).
			Label("appmana.com/tb-chain-name", chainName).
			Label("appmana.com/tb-chain-index", strconv.Itoa(i)).
			Label(corev1.LabelHostname, hostname).
			Ready().
			Obj()
	}
	return out
}

// chainLevels is the canonical 3-level chain topology used in these tests.
var chainLevels = []string{
	"appmana.com/tb-chain-name",
	"appmana.com/tb-chain-index",
	corev1.LabelHostname,
}

// chainOrdered marks the chain-index level as Ordered.
var chainOrdered = []bool{false, true, false}

// TestOrderedSnapshot_ChildrenSortedNumerically verifies the snapshot
// builds chain-index children in 0,1,2,...,11 order at ordered levels —
// not the lex order 0,1,10,11,2,3,... that the upstream string compare
// would yield.
func TestOrderedSnapshot_ChildrenSortedNumerically(t *testing.T) {
	_, log := utiltesting.ContextWithLog(t)
	nodes := chainNodes("primary", 12)

	s := newTASFlavorSnapshot(log, "tb-chain", chainLevels, nil)
	s.setLevelOrdered(chainOrdered)
	for i := range nodes {
		s.addNode(newNodeInfo(&nodes[i]))
	}
	s.initialize()

	// One chain-name root, with 12 chain-index children.
	if len(s.roots) != 1 {
		t.Fatalf("roots = %d, want 1", len(s.roots))
	}
	var chainRoot *domain
	for _, r := range s.roots {
		chainRoot = r
	}
	if len(chainRoot.children) != 12 {
		t.Fatalf("chain-index children count = %d, want 12", len(chainRoot.children))
	}

	for i, child := range chainRoot.children {
		got := child.levelValues[1]
		want := strconv.Itoa(i)
		if got != want {
			t.Errorf("children[%d].chain-index = %q, want %q (numeric ordering broken — got lex order?)",
				i, got, want)
		}
	}
}

// TestOrderedSnapshot_BuildAssignmentNumericOrder feeds buildAssignment a
// shuffled list of chain-index domains and asserts the emitted assignment
// has them in numeric order. This is the property the ungater reads to map
// rank N to chain-index N within the picked range.
func TestOrderedSnapshot_BuildAssignmentNumericOrder(t *testing.T) {
	_, log := utiltesting.ContextWithLog(t)
	nodes := chainNodes("primary", 12)

	s := newTASFlavorSnapshot(log, "tb-chain", chainLevels, nil)
	s.setLevelOrdered(chainOrdered)
	for i := range nodes {
		s.addNode(newNodeInfo(&nodes[i]))
	}
	s.initialize()

	// Feed buildAssignment all 12 leaf domains in REVERSE order with
	// state=1 each (one pod per host). The final assignment should be in
	// numeric order regardless of input order.
	leaves := make([]*domain, 0, 12)
	for _, leaf := range s.leaves {
		d := &leaf.domain
		d.state = 1
		leaves = append(leaves, d)
	}
	// Sort in DESCENDING chain-index to exercise the sort within
	// buildAssignment.
	for i, j := 0, len(leaves)-1; i < j; i, j = i+1, j-1 {
		leaves[i], leaves[j] = leaves[j], leaves[i]
	}

	asg := s.buildAssignment(leaves)
	if asg == nil {
		t.Fatalf("buildAssignment returned nil")
	}

	// With isLowestLevelNode=true, the assignment elides upper levels and
	// keeps just the hostname — domain values are ["h0"], ["h1"], etc.
	// The order must follow chain-index, which means hostnames in 0..11
	// order. (chainNodes labels host hN with chain-index=N.)
	if len(asg.Domains) != 12 {
		t.Fatalf("got %d domains, want 12", len(asg.Domains))
	}
	for i, d := range asg.Domains {
		want := fmt.Sprintf("h%d", i)
		got := d.Values[len(d.Values)-1]
		if got != want {
			t.Errorf("position %d: got hostname %q, want %q", i, got, want)
		}
	}
}

// TestOrderedSnapshot_UnorderedDefaultPreservesLex ensures we haven't
// broken upstream behaviour when Ordered is not set.  Same chain-index
// labels but with the level marked unordered → the assignment comes out
// in lex order (0,1,10,11,2,...).  This is the regression guard: default
// false must reproduce original behaviour exactly.
func TestOrderedSnapshot_UnorderedDefaultPreservesLex(t *testing.T) {
	_, log := utiltesting.ContextWithLog(t)
	nodes := chainNodes("primary", 12)

	s := newTASFlavorSnapshot(log, "tb-chain", chainLevels, nil)
	// No setLevelOrdered call → all levels default to unordered.
	for i := range nodes {
		s.addNode(newNodeInfo(&nodes[i]))
	}
	s.initialize()

	leaves := make([]*domain, 0, 12)
	for _, leaf := range s.leaves {
		d := &leaf.domain
		d.state = 1
		leaves = append(leaves, d)
	}
	asg := s.buildAssignment(leaves)
	if asg == nil || len(asg.Domains) != 12 {
		t.Fatalf("got %d domains, want 12", len(asg.Domains))
	}
	// Lex order on hostnames "h0".."h11": h0, h1, h10, h11, h2, h3, ..., h9.
	wantLex := []string{"h0", "h1", "h10", "h11", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"}
	for i, d := range asg.Domains {
		got := d.Values[len(d.Values)-1]
		if got != wantLex[i] {
			t.Errorf("position %d: got %q, want %q (default unordered should preserve lex behaviour)",
				i, got, wantLex[i])
		}
	}
}

// TestCompareDomainLevelValues exercises the comparator directly across
// several edge cases — Ordered vs unordered levels, mixed values, lengths.
func TestCompareDomainLevelValues(t *testing.T) {
	cases := []struct {
		name     string
		ordered  []bool
		a, b     []string
		wantSign int
	}{
		{
			name:     "ordered level: numeric beats lex (10 > 2)",
			ordered:  []bool{false, true},
			a:        []string{"primary", "10"},
			b:        []string{"primary", "2"},
			wantSign: +1,
		},
		{
			name:     "ordered level: 2 < 10 numerically",
			ordered:  []bool{false, true},
			a:        []string{"primary", "2"},
			b:        []string{"primary", "10"},
			wantSign: -1,
		},
		{
			name:     "unordered level falls back to lex (10 < 2 lex)",
			ordered:  []bool{false, false},
			a:        []string{"primary", "10"},
			b:        []string{"primary", "2"},
			wantSign: -1,
		},
		{
			name:     "ordered level with non-numeric falls back to lex",
			ordered:  []bool{false, true},
			a:        []string{"primary", "abc"},
			b:        []string{"primary", "abd"},
			wantSign: -1,
		},
		{
			name:     "first ordered then unordered",
			ordered:  []bool{true, false},
			a:        []string{"3", "z"},
			b:        []string{"10", "a"},
			wantSign: -1, // 3 < 10 numerically; 'z' vs 'a' irrelevant
		},
		{
			name:     "tie at all levels",
			ordered:  []bool{true, true},
			a:        []string{"5", "7"},
			b:        []string{"5", "7"},
			wantSign: 0,
		},
		{
			name:     "longer slice tied prefix",
			ordered:  []bool{false, true},
			a:        []string{"primary", "5"},
			b:        []string{"primary", "5", "h0"},
			wantSign: -1, // shorter slice loses
		},
		{
			name:     "no ordered flags configured (legacy path)",
			ordered:  nil,
			a:        []string{"a", "10"},
			b:        []string{"a", "2"},
			wantSign: -1, // pure lex
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := &TASFlavorSnapshot{}
			s.levelOrdered = c.ordered

			got := s.compareDomainLevelValues(c.a, c.b)
			gotSign := 0
			switch {
			case got < 0:
				gotSign = -1
			case got > 0:
				gotSign = +1
			}
			if gotSign != c.wantSign {
				t.Errorf("got %d (sign %d), want sign %d", got, gotSign, c.wantSign)
			}
		})
	}
}

// TestOrderedSnapshot_HasOrderedLevels covers the fast-path predicate.
func TestOrderedSnapshot_HasOrderedLevels(t *testing.T) {
	cases := []struct {
		name string
		in   []bool
		want bool
	}{
		{"nil", nil, false},
		{"all false", []bool{false, false, false}, false},
		{"middle true", []bool{false, true, false}, true},
		{"first true", []bool{true, false}, true},
		{"all true", []bool{true, true}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := &TASFlavorSnapshot{levelOrdered: c.in}
			if got := s.hasOrderedLevels(); got != c.want {
				t.Errorf("hasOrderedLevels() = %v, want %v", got, c.want)
			}
		})
	}
}
