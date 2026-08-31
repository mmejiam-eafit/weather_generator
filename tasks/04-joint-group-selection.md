# Task 04 — Joint-group selection

**Depends on:** — · **Blocks:** 05 (shares this task's experiment)
**Open question:** `docs/open_questions.md` §3.2 — OPEN
**Primary source docs:** `docs/correlated_variables.md` §4, §5

## 1. What question this will try to answer

Which variables belong in a joint DMD group, and how many groups are
needed? No general algorithm for group membership is given anywhere.
Growing a group (e.g. adding the peak-wave triplet to capture $S_{nl}$'s
energy transfer beyond the wind-sea partition) increases the joint state's
dimension and hence the DMD rank and data requirements, risking an
underdetermined or overfit joint operator for a fixed training window;
keeping groups small and conservative avoids that risk but reproduces the
independent-fit blind spot for whatever coupling falls outside the chosen
group. This is a modeling/hyperparameter choice to be made empirically.

## 2. Steps to answer it

1. Fit `correlated_variables.md` §4's construction exactly as specified:
   independent per-variable Tucker cores (`tensorly.decomposition.partial_tucker`,
   masked) for the wind-sea generation group's five members
   (`u_wind, v_wind` from `was`/`wad_come_from`; `u_wsea, v_wsea` from
   `wsh`/`wsd`; `T` from `wsp`), vectorize each member's per-timestep
   coefficient grid, concatenate into one joint state, and fit one joint
   DMD (`train.dmd.dmd_fit`) on it.
2. Per §5's diagnostic: partition the fitted joint operator into blocks
   aligned with each member and check whether off-diagonal blocks (e.g.
   the block mapping current wind-coefficient state to next-step
   wind-sea-coefficient state) are **non-negligible** relative to the
   diagonal blocks. Non-negligible off-diagonal structure is direct,
   fitted evidence the group is well-chosen; negligible structure means
   the group was drawn too widely for no benefit (§3.2's own stated
   diagnostic).
3. Repeat with the extended group (`correlated_variables.md` §4.1's second
   candidate: wind-sea group ∪ `u_pwave, v_pwave, T` from the peak-wave
   triplet) and compare — does the extra group size actually earn
   non-negligible new off-diagonal structure, or does it just grow
   $r_{dmd}$'s data requirement for no benefit?
4. **Decision rule:** prefer the smallest group whose off-diagonal blocks
   are non-negligible; flag any group whose singular-value spectrum shows
   no clear elbow or whose off-diagonal blocks are unstable across
   refits/bootstrap windows as "grown past what the available $T$ can
   support" (§5's rank/data-tradeoff diagnostic).

## 3. Notebook(s) needed

- `notebooks/04_joint_group_selection.ipynb` — loads the wind-sea
  generation group's five source fields, fits independent per-member
  Tucker cores, builds the joint state, and fits joint DMD. The
  block-structure comparison (step 2–3) is marked as the concrete next
  cells to add once run against real data. Also used by Task 05 (§3 there).

## 4. Conclusion

_(Blank. Fill in only after the notebook has been run against real archive
data — measured off-diagonal block strength for both candidate groups,
whether it changes `open_questions.md` §3.2's status tag, and the group
recommendation this task should produce.)_
