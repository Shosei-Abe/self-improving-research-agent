-- invariant: 1 ≤ citationsPerParagraph ≤ 10
-- This mirrors the 'citationsPerParagraph_realistic' check in App.jsx L1557-1560
-- and metrics.py's range enforcement.

theorem citations_range (c : Nat) (h1 : c ≥ 1) (h2 : c ≤ 10) :
    c ≥ 1 ∧ c ≤ 10 := by
  exact ⟨h1, h2⟩

theorem citations_nontrivial_mutation (old new : Nat)
    (h_range : new ≥ 1 ∧ new ≤ 10)
    (h_diff : old ≠ new) :
    new ≥ 1 ∧ new ≤ 10 ∧ old ≠ new := by
  exact ⟨h_range.1, h_range.2, h_diff⟩

#check citations_range
#check citations_nontrivial_mutation
