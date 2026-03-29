# modqldpc/runtime/layer_exec.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Any

from .outcomes import OutcomeModel
from .frame_policy import FrameState, FrameUpdatePolicy, AxisRewriteLog
from .pauli_ops import PauliAxis

from ..core.types import PauliRotation  
from ..lowering.ir import K_FRAME_UPDATE, K_MEAS_MAGIC_X, K_MEAS_PARITY_PZ, ExecNode, ClassicalKey 


@dataclass
class LayerExecutionResult:
    layer: int
    frame_after: FrameState
    # what we generated in this layer
    produced_bits: Dict[str, int] = field(default_factory=dict)
    # rewritten next-layer rotations
    next_rotations_effective: List[PauliRotation] = field(default_factory=list)
    rewrite_log: List[AxisRewriteLog] = field(default_factory=list)
    # execution log
    events: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0


def _keys_to_names(keys: List[ClassicalKey]) -> List[str]:
    return [k.name for k in keys]


class LayerExecutor:
    def __init__(self, *, outcome_model: OutcomeModel, frame_policy: FrameUpdatePolicy):
        self.outcome_model = outcome_model
        self.frame_policy = frame_policy

    def execute_layer(
        self,
        *,
        layer: int,
        dag: Any,  # your Graph/ExecDAG with .nodes/.pred/.succ
        schedule: Any,  # your Schedule with meta["entries"] start/end
        frame_in: FrameState,
        next_layer_rotations: List[PauliRotation],
    ) -> LayerExecutionResult:

        # Copy frame so we don't mutate caller’s object
        st = FrameState(
            bits=dict(frame_in.bits),
            clifford_pi4_generators=list(frame_in.clifford_pi4_generators),
            pauli_byproducts=list(frame_in.pauli_byproducts),
        )

        entries = schedule.meta.get("entries", {})
        if not entries:
            raise ValueError("Schedule.meta['entries'] missing; need start/end intervals.")

        # build end-time buckets
        end_at: Dict[int, List[str]] = {}
        for nid, se in entries.items():
            n = dag.nodes[nid]
            dur = int(getattr(n, "duration", 1))
            if dur > 0:
                end_at.setdefault(int(se["end"]), []).append(nid)
        # track completion for readiness
        done: Set[str] = set()
        indeg_left: Dict[str, int] = {nid: len(dag.pred.get(nid, set())) for nid in dag.nodes}
        ready_zero: Set[str] = set()
        # print(f"Initial indeg_left: {indeg_left}")
        # helper: when node becomes unblocked, if duration==0 we can run immediately
        def maybe_enqueue_zero(nid: str):
            n: ExecNode = dag.nodes[nid]
            if int(getattr(n, "duration", 1)) == 0 and nid not in done:
                ready_zero.add(nid)

        # Initialize any zero-duration nodes that already have indeg 0
        for nid, d in indeg_left.items():
            if d == 0:
                maybe_enqueue_zero(nid)

        # Iterate time in order
        times = sorted(end_at.keys())
        events: List[Dict[str, Any]] = []
        produced_bits: Dict[str, int] = {}

        def complete_node(nid: str, t: int):
            n: ExecNode = dag.nodes[nid]
            # print(f"Completing node {nid} kind {n.kind} at time {t}")
            # If measurement node: sample bits for all produced keys
            if n.kind in (K_MEAS_PARITY_PZ, K_MEAS_MAGIC_X):
                for k in n.produces:
                    b = int(self.outcome_model.sample_bit(k))
                    st.bits[k.name] = b
                    # print(f"Measured {k.name}={b} at node {nid} kind {n.kind}")
                    produced_bits[k.name] = b
                events.append({"t": t, "nid": nid, "kind": n.kind, "produced": {k: st.bits[k] for k in _keys_to_names(n.produces)}})

            # If frame update node: consume exactly one key (your depends_on) and apply rule
            elif n.kind == K_FRAME_UPDATE:
                 # your node_frame_update should set:
                #  - meta["update_kind"] in {"clifford_pi4","pauli"}
                #  - meta["axis"] or store axis in meta or another field
                update_kind = n.meta.get("update_kind")
                axis_dict = n.meta.get("axis")
                if update_kind is None or axis_dict is None:
                    raise ValueError(f"frame_update node {nid} missing meta.update_kind or meta.axis")

                axis = PauliAxis(sign=int(axis_dict["sign"]), tensor=str(axis_dict["tensor"]))

                if not n.consumes:
                    raise ValueError(f"frame_update node {nid} must consume 1 ClassicalKey")
                dep = n.consumes[0].name
                # print(st.bits)
                # print(f"Calling update frame {dep}->{update_kind}")
                if dep not in st.bits:
                    raise RuntimeError(
                        f"Frame update node {nid} requires bit '{dep}', but it is not available yet. "
                        f"Check same-time ordering / zero-duration execution."
                    )
                bit = st.bits[dep]
                # print(f"Calling update frame {dep}->{bit}->{update_kind}")

                self.frame_policy.apply_frame_update(update_kind=update_kind, bit=bit, axis=axis, st=st)
                events.append({"t": t, "nid": nid, "kind": n.kind, "update_kind": update_kind, "bit": bit, "axis": axis.tensor})

            else:
                # physical plumbing nodes
                events.append({"t": t, "nid": nid, "kind": n.kind})

            # mark done and unlock successors
            done.add(nid)
            for ch in dag.succ.get(nid, set()):
                indeg_left[ch] -= 1
                if indeg_left[ch] == 0:
                    maybe_enqueue_zero(ch)

        def drain_zero(t: int):
            # execute any ready zero-duration nodes until fixed point
            # deterministic order for stability
            changed = True
            while changed:
                changed = False
                for nid in sorted(list(ready_zero)):
                    if nid in done:
                        ready_zero.remove(nid)
                        continue
                    # ensure all preds done
                    if any(p not in done for p in dag.pred.get(nid, set())):
                        continue
                    ready_zero.remove(nid)
                    complete_node(nid, t)
                    changed = True

        for t in times:
            # complete all nodes whose schedule interval ends at t
            for nid in sorted(end_at[t]):
                # print(f"Time {t}: completing node {nid} kind {dag.nodes[nid].kind}")
                if nid in done:
                    continue
                complete_node(nid, t)

            # after physical completions at time t, run any 0-duration updates now
            drain_zero(t)

        # Final sanity: all nodes must be done
        if len(done) != len(dag.nodes):
            missing = sorted(set(dag.nodes.keys()) - done)[:20]
            raise RuntimeError(f"Layer execution incomplete. Missing {len(dag.nodes)-len(done)} nodes, e.g. {missing}")

        # Rewrite next layer rotations using accumulated frame
        rewrite_log: List[AxisRewriteLog] = []
        next_eff: List[PauliRotation] = []
        for r in next_layer_rotations:
            Q = PauliAxis(sign=1 if r.angle>=0 else -1, tensor=r.axis.lstrip("+-"))
            ang = r.angle
            Q2, ang2, reason = self.frame_policy.rewrite_axis(Q, ang, st)
            # print(f"Before: {Q.tensor}, After: {Q2.tensor}")
            changed_support = (set(i for i,c in enumerate(Q.tensor) if c!="I") != set(i for i,c in enumerate(Q2.tensor) if c!="I"))
            if Q2 != Q or ang2 != ang:
                rewrite_log.append(
                    AxisRewriteLog(
                        ridx=r.idx,
                        before=Q, after=Q2,
                        angle_before=ang, angle_after=ang2,
                        changed_support=changed_support,
                        reason=reason,
                    )
                )
            next_eff.append(PauliRotation(axis=Q2, angle=ang2, source=r.source, idx=r.idx))

        return LayerExecutionResult(
            layer=layer,
            frame_after=st,
            produced_bits=produced_bits,
            next_rotations_effective=next_eff,
            rewrite_log=rewrite_log,
            events=events,
            depth=max(times) if times else 0,
        )