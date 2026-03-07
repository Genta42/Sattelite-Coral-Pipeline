# Independent correctness checks for tilings of a 2×n board with:
# - dominoes (2 cells)
# - L-trominoes (3 cells in a 2×2 missing one corner)
# Rotations allowed. No cutting.
# This file uses profile-DP for the main answer and a brute-force solver as an independent checker for small n.

from functools import lru_cache

def count_by_profile_dp(n: int) -> int:
    """
    Ground-truth DP using column profile (bitmask) recursion.
    This does NOT use the closed-form recurrence.
    State: (col, mask) where mask is occupancy of current column's 2 cells:
      bit0 = top filled, bit1 = bottom filled.
    We fill the current column left-to-right, allowing placements that may extend to next column.
    """

    # helper: try to fill within column 'c' given current mask and next_mask (cells already occupied in next col)
    @lru_cache(None)
    def solve(c: int, mask: int) -> int:
        if c == n:
            return 1 if mask == 0 else 0  # no next column exists
        return fill_column(c, mask, 0)

    @lru_cache(None)
    def fill_column(c: int, mask: int, next_mask: int) -> int:
        # If current column fully filled, move to next column
        if mask == 0b11:
            return solve(c + 1, next_mask)

        # Find first empty cell in current column
        if (mask & 0b01) == 0:
            r = 0  # top cell empty
        else:
            r = 1  # bottom cell empty

        total = 0

        def set_bit(m, bit):
            return m | (1 << bit)

        # 1) Vertical domino (only if both cells empty in this column)
        if r == 0 and (mask & 0b11) == 0:
            total += fill_column(c, 0b11, next_mask)

        # 2) Horizontal domino from (r,c) to (r,c+1)
        # Requires that c+1 exists and the corresponding cell in next column not already occupied.
        if c + 1 < n:
            if r == 0 and (next_mask & 0b01) == 0:
                total += fill_column(c, set_bit(mask, 0), set_bit(next_mask, 0))
            if r == 1 and (next_mask & 0b10) == 0:
                total += fill_column(c, set_bit(mask, 1), set_bit(next_mask, 1))

        # 3) L-trominoes: they always occupy 3 cells in a 2×2 block spanning columns c and c+1.
        # So we need c+1 < n.
        if c + 1 < n:
            # There are 4 orientations (missing one of the 4 cells in the 2×2).
            # Represent the 2×2 cells as:
            #  current col: (top=c0, bottom=c1) => bits 0,1 in mask
            #  next col:    (top=n0, bottom=n1) => bits 0,1 in next_mask
            #
            # For each missing corner, try placing the other 3 cells if they are all free.

            # Missing current-top: occupy current-bottom, next-top, next-bottom
            if (mask & 0b10) == 0 and (next_mask & 0b01) == 0 and (next_mask & 0b10) == 0:
                total += fill_column(c, set_bit(mask, 1), next_mask | 0b11)

            # Missing current-bottom: occupy current-top, next-top, next-bottom
            if (mask & 0b01) == 0 and (next_mask & 0b01) == 0 and (next_mask & 0b10) == 0:
                total += fill_column(c, set_bit(mask, 0), next_mask | 0b11)

            # Missing next-top: occupy current-top, current-bottom, next-bottom
            if (mask & 0b01) == 0 and (mask & 0b10) == 0 and (next_mask & 0b10) == 0:
                total += fill_column(c, mask | 0b11, set_bit(next_mask, 1))

            # Missing next-bottom: occupy current-top, current-bottom, next-top
            if (mask & 0b01) == 0 and (mask & 0b10) == 0 and (next_mask & 0b01) == 0:
                total += fill_column(c, mask | 0b11, set_bit(next_mask, 0))

        return total

    return solve(0, 0)



def count_by_bruteforce(n: int) -> int:
    """
    Brute-force backtracking tiler for a 2×n board.
    This is exponential and intended only as an independent correctness checker
    for small n (e.g., n <= 9).

    Board indexing:
      cell (r, c) maps to bit idx = 2*c + r, where r=0 is top, r=1 is bottom.
    """

    total_cells = 2 * n

    @lru_cache(None)
    def rec(mask: int) -> int:
        # mask bit = 1 means occupied
        if mask == (1 << total_cells) - 1:
            return 1

        # find first empty cell
        i = 0
        while (mask >> i) & 1:
            i += 1

        c = i // 2
        r = i % 2

        def occ(m: int, rr: int, cc: int) -> bool:
            return (m >> (2 * cc + rr)) & 1 == 1

        def set_occ(m: int, rr: int, cc: int) -> int:
            return m | (1 << (2 * cc + rr))

        ways = 0

        # --- Domino placements ---
        # Vertical domino (fills both cells in the same column)
        if r == 0 and not occ(mask, 1, c):
            m2 = set_occ(set_occ(mask, 0, c), 1, c)
            ways += rec(m2)

        # Horizontal domino (fills (r,c) and (r,c+1))
        if c + 1 < n and not occ(mask, r, c + 1):
            m2 = set_occ(set_occ(mask, r, c), r, c + 1)
            ways += rec(m2)

        # --- L-tromino placements ---
        # L-tromino always fits inside a 2×2 block spanning columns c and c+1.
        if c + 1 < n:
            # cells in the 2×2 block:
            # (0,c), (1,c), (0,c+1), (1,c+1)
            cells = [(0, c), (1, c), (0, c + 1), (1, c + 1)]

            # Four orientations = choose which corner is missing.
            for miss in range(4):
                used = [cells[j] for j in range(4) if j != miss]

                # must include the anchor cell (r,c)
                if (r, c) not in used:
                    continue

                # check all used cells are free
                ok = True
                for rr, cc in used:
                    if occ(mask, rr, cc):
                        ok = False
                        break
                if not ok:
                    continue

                m2 = mask
                for rr, cc in used:
                    m2 = set_occ(m2, rr, cc)
                ways += rec(m2)

        return ways

    return rec(0)


def run_checks(max_n: int = 13) -> None:
    ok = True
    # Brute force is exponential; keep it small for validation.
    brute_max = min(max_n, 9)

    for n in range(max_n + 1):
        a = count_by_profile_dp(n)

        if n <= brute_max:
            b = count_by_bruteforce(n)
            status = "OK" if a == b else "MISMATCH"
            print(f"n={n:2d}  profile_dp={a:8d}  bruteforce={b:8d}  {status}")
            if a != b:
                ok = False
        else:
            # For large n, we only print the DP result (brute force too slow).
            print(f"n={n:2d}  profile_dp={a:8d}  bruteforce=   (skip)  SKIPPED")

    print("\nOverall:", "ALL CHECKS PASSED ✅" if ok else "FAILED ❌")


if __name__ == "__main__":
    run_checks(13)
    print("\nAnswer for 2×13 =", count_by_profile_dp(13))