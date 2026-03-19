# modqldpc/scheduling/ownership.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

_INF: int = 10 ** 9  # sentinel: ownership end not yet determined


@dataclass
class _Interval:
    start: int
    end: int   # _INF until the bounding node (PZ or Xm) is scheduled
    cid: int


@dataclass
class BlockOwnershipTracker:
    """
    Tracks which rotation-component exclusively owns each block and coupler
    across the multi-node span of a gadget measurement.

    Three scopes are enforced simultaneously:

      (1) Preparation lock
            Each block is owned by the rotation from the moment its first
            init_pivot is scheduled until the block's measurement stage ends:
              - single-block rotation  → until meas_magic_X end
              - multi-block magic block → until meas_magic_X end
              - multi-block non-magic   → until meas_parity_PZ end

      (2) Route lock
            From interblock_link start to meas_parity_PZ end, every block and
            coupler on the chosen route is exclusively reserved.

      (3) Magic-block lock
            The magic block stays locked from interblock_link start (or first
            init for single-block) through meas_magic_X end.

    Usage pattern
    -------------
    On init_pivot(block B, cid):
        ownership.claim_block(B, t, cid)          # end=INF, to be filled later

    On interblock_link(route_blocks, route_couplers, cid):
        ownership.claim_block(rb, t, cid)         # for each intermediate block
        ownership.claim_coupler(rc, t, cid)       # for each route coupler

    On meas_parity_PZ(pz_end, cid):
        ownership.update_block_end(b, pz_end, cid)    # for each non-magic participant block
        ownership.update_block_end(rb, pz_end, cid)   # for each intermediate route block
        ownership.update_coupler_end(c, pz_end, cid)  # for each route coupler

    On meas_magic_X(xm_end, cid):
        ownership.update_block_end(magic_block, xm_end, cid)
    """

    # At most one active interval per block / coupler.
    block_intervals: Dict[int, _Interval] = field(default_factory=dict)
    coupler_intervals: Dict[str, _Interval] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Block operations                                                     #
    # ------------------------------------------------------------------ #

    def can_claim_block(self, block: int, t_start: int, cid: int) -> bool:
        """
        Returns True if component `cid` may claim `block` starting at `t_start`.
        Conditions:
          - block has no active interval, OR
          - the active interval belongs to the same component (safe, e.g. link
            re-claiming a terminal block already held by the same rotation), OR
          - the active interval has expired (end <= t_start).
        """
        iv = self.block_intervals.get(block)
        if iv is None:
            return True
        if iv.cid == cid:
            return True
        return iv.end <= t_start

    def block_conflict_info(self, block: int, t_start: int, cid: int) -> Optional[str]:
        """Human-readable reason why block cannot be claimed (None if OK)."""
        iv = self.block_intervals.get(block)
        if iv is None or iv.cid == cid or iv.end <= t_start:
            return None
        end_str = "INF" if iv.end == _INF else str(iv.end)
        return (
            f"block {block} owned by cid={iv.cid} "
            f"[{iv.start}, {end_str})  (want cid={cid} @ t={t_start})"
        )

    def claim_block(self, block: int, t_start: int, cid: int) -> None:
        """Claim `block` for `cid` starting at `t_start` (end=INF until updated)."""
        self.block_intervals[block] = _Interval(start=t_start, end=_INF, cid=cid)

    def update_block_end(self, block: int, end: int, cid: int) -> None:
        """Finalise the end time for `cid`'s current interval on `block`."""
        iv = self.block_intervals.get(block)
        if iv is not None and iv.cid == cid and iv.end == _INF:
            iv.end = end

    # ------------------------------------------------------------------ #
    # Coupler operations                                                   #
    # ------------------------------------------------------------------ #

    def can_claim_coupler(self, coupler: str, t_start: int, cid: int) -> bool:
        iv = self.coupler_intervals.get(coupler)
        if iv is None:
            return True
        if iv.cid == cid:
            return True
        return iv.end <= t_start

    def coupler_conflict_info(self, coupler: str, t_start: int, cid: int) -> Optional[str]:
        iv = self.coupler_intervals.get(coupler)
        if iv is None or iv.cid == cid or iv.end <= t_start:
            return None
        end_str = "INF" if iv.end == _INF else str(iv.end)
        return (
            f"coupler {coupler!r} owned by cid={iv.cid} "
            f"[{iv.start}, {end_str})  (want cid={cid} @ t={t_start})"
        )

    def claim_coupler(self, coupler: str, t_start: int, cid: int) -> None:
        self.coupler_intervals[coupler] = _Interval(start=t_start, end=_INF, cid=cid)

    def update_coupler_end(self, coupler: str, end: int, cid: int) -> None:
        iv = self.coupler_intervals.get(coupler)
        if iv is not None and iv.cid == cid and iv.end == _INF:
            iv.end = end

    # ------------------------------------------------------------------ #
    # Debug helpers                                                        #
    # ------------------------------------------------------------------ #

    def print_state(self, t: int, label: str = "") -> None:
        tag = f"[OwnershipTracker{' ' + label if label else ''} @ t={t}]"
        active_blocks = {
            b: iv for b, iv in self.block_intervals.items() if iv.end > t
        }
        active_couplers = {
            c: iv for c, iv in self.coupler_intervals.items() if iv.end > t
        }
        if not active_blocks and not active_couplers:
            print(f"  {tag} all free")
            return
        print(f"  {tag}")
        for b, iv in sorted(active_blocks.items()):
            end_str = "INF" if iv.end == _INF else str(iv.end)
            print(f"    block   {b:4d}  cid={iv.cid:3d}  [{iv.start}, {end_str})")
        for c, iv in sorted(active_couplers.items()):
            end_str = "INF" if iv.end == _INF else str(iv.end)
            print(f"    coupler {c!r:20s}  cid={iv.cid:3d}  [{iv.start}, {end_str})")

    def get_all_intervals(self) -> Tuple[
        Dict[int, List[Tuple[int, int, int]]],
        Dict[str, List[Tuple[int, int, int]]],
    ]:
        """
        Returns (block_hist, coupler_hist) for external validation.
        Each entry is (start, end, cid).  end=_INF means the tracker was
        never finalized (schedule incomplete or bug).
        """
        bh: Dict[int, List[Tuple[int, int, int]]] = {}
        for b, iv in self.block_intervals.items():
            bh.setdefault(b, []).append((iv.start, iv.end, iv.cid))
        ch: Dict[str, List[Tuple[int, int, int]]] = {}
        for c, iv in self.coupler_intervals.items():
            ch.setdefault(c, []).append((iv.start, iv.end, iv.cid))
        return bh, ch
