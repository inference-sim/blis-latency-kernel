// Package price holds the cost laws the kernel composes: the GEMM efficiency ramp, the
// cross-node collective scaling, and the resource maximum.
//
// Each law is a function with no state, tested against the properties it must have rather
// than only against values. A law that is monotone where it should be, bounded where it
// should be, and exact at its defining points is one a reader can reason about; a law
// checked only at sampled points can be wrong between them in ways no example reveals.
package price

import "math"

// Efficiency returns the fraction of peak a matmul achieves at m rows.
//
//	eff(m) = epsMax * m / (m + mHalf)
//
// Monotone increasing in m, bounded above by epsMax, zero at m = 0, and exactly epsMax/2
// at m = mHalf — which is what makes mHalf's name meaningful. A cost model that used peak
// instead would be optimistic by 1/eff, which at a 512-token batch is about 1.6x.
func Efficiency(m float64, epsMax, mHalf float64) float64 {
	if m <= 0 || epsMax <= 0 {
		return 0
	}
	if mHalf <= 0 {
		return epsMax
	}
	return epsMax * m / (m + mHalf)
}

// Span returns the factor by which a collective's cost rises when its ranks span more
// than one node.
//
//	span(cross, total, ratio) = 1 + (ratio - 1) * cross / total
//
// where ratio is intra-node bandwidth over inter-node, cross is the number of hops that
// leave a node and total the hops in the collective. Four properties hold, and a cost
// model that violated any of them would be wrong in a direction arithmetic alone would
// not reveal:
//
//   - monotone in ratio: a worse fabric never lowers the cost;
//   - never below 1: spanning is never cheaper than staying on-node;
//   - exactly 1 when cross is 0, which is the single-node and the in-rack NVLink case;
//   - a degenerate ratio resolves to 1 rather than propagating, so an absent or malformed
//     bandwidth does not become a NaN step time.
func Span(cross, total int, ratio float64) float64 {
	if cross <= 0 || total <= 0 {
		return 1
	}
	if math.IsNaN(ratio) || math.IsInf(ratio, 0) || ratio <= 1 {
		return 1
	}
	return 1 + (ratio-1)*float64(cross)/float64(total)
}

// RingSpan returns the scaling for a ring collective — an all-reduce, or the
// all-gather and reduce-scatter pair — over group ranks placed gPerNode to a node.
//
// A ring reduces as it travels: an intra-node reduce-scatter and all-gather over the
// ranks on a node, then an inter-node all-reduce of the already-reduced 1/gPerNode chunk.
// Per-rank bytes match a flat ring exactly, but only a 1/gPerNode share crosses the
// fabric, so the cross-hop count is nodes-1 of group-1 hops rather than all of them.
//
// Charging a ring the flat-ring cost instead overstates it by up to the raw link ratio —
// nearly six times at a 2048-token batch over a 9x fabric — which is the error a
// bandwidth-only model makes.
func RingSpan(group, gPerNode int, ratio float64) float64 {
	if group <= 1 || gPerNode <= 0 {
		return 1
	}
	nodes := group / gPerNode
	if group%gPerNode != 0 {
		// The hierarchical decomposition is exact only when the node size divides the
		// group. An uneven split has no such decomposition, so the honest answer is the
		// flat-ring bound rather than a formula that does not apply.
		return FlatSpan(group, gPerNode, ratio)
	}
	return Span(nodes-1, group-1, ratio)
}

// All2AllSpan returns the scaling for a genuine point-to-point all-to-all.
//
// Unlike a ring, nothing reduces on the way: every rank must deliver data to every other,
// and group-gPerNode of its group-1 peers sit off-node. So a far larger share of the
// traffic crosses the fabric — 5.3x against a ring's 1.5x at sixteen ranks over two
// eight-GPU nodes on a 9x fabric.
//
// Which of the two applies is a property of the backend rather than of the model: an
// all-gather-family MoE backend moves ring-shaped volume even though the primitive is
// named all-to-all.
func All2AllSpan(group, gPerNode int, ratio float64) float64 {
	if group <= 1 || gPerNode <= 0 {
		return 1
	}
	return Span(group-gPerNode, group-1, ratio)
}

// FlatSpan returns the scaling for a collective that does not decompose hierarchically:
// every hop pays the slowest link. It is the bound used where a group's placement admits
// no clean decomposition.
func FlatSpan(group, gPerNode int, ratio float64) float64 {
	if group <= 1 || gPerNode <= 0 || group <= gPerNode {
		return 1
	}
	return Span(group-1, group-1, ratio)
}

// CollectiveTime returns the time to move bytes through one collective.
//
//	max(floor + bytes/transitionRate, bytes/peakRate)
//
// Three parameters rather than two, because two do not fit the data. The obvious form is
// max(bytes/peak, floor) — the floor is the cost of the smallest transfer, a larger
// transfer subsumes it, so take whichever binds. That reasoning is sound at both ends of
// the curve and wrong between them: measured against AISimulate's NCCL sweeps it
// understates an 8-rank H200 all-reduce by up to 3.2x and an all-gather by up to 3.8x,
// because a collective reaches its asymptotic bandwidth only at messages far larger than
// one forward pass produces. The region where it is worst, roughly 64 KiB to 8 MiB per
// collective, is exactly where a prefill batch's collectives land.
//
// So the setup cost IS additive over the transition, and the asymptote is a ceiling on
// achievable bandwidth rather than the rate to use. Over all 136 measured configurations
// this form takes the median geometric error from 1.34x to 1.17x and the worst case from
// 1.83x to 1.51x. The improvement survives transfer between rank counts — fit the rate at
// 4 ranks, price 8 — so it is structure and not curve-hugging.
//
// A zero or absent transition rate falls back to the two-parameter form, so a coefficient
// set predating the third parameter still prices, less accurately and visibly so.
func CollectiveTime(bytes, transitionRate, peakRate, floorSeconds float64) float64 {
	if bytes <= 0 {
		return floorSeconds
	}
	if peakRate <= 0 {
		return math.Inf(1)
	}
	atPeak := bytes / peakRate
	if transitionRate <= 0 {
		return math.Max(floorSeconds, atPeak)
	}
	return math.Max(floorSeconds+bytes/transitionRate, atPeak)
}

// FloorAndRate is the two-parameter form, kept for transfers that have no measured
// transition rate: a tier read, a PD transfer over a fabric. Those are point-to-point
// rather than collective, so the shape the third parameter corrects does not apply — a
// single link's bandwidth does not ramp with message size the way a multi-rank
// collective's effective bandwidth does.
func FloorAndRate(bytes float64, bytesPerSecond, floorSeconds float64) float64 {
	if bytes <= 0 {
		return floorSeconds
	}
	if bytesPerSecond <= 0 {
		return math.Inf(1)
	}
	return math.Max(floorSeconds, bytes/bytesPerSecond)
}
