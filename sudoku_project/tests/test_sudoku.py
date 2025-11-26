import unittest
from src.sudoku_project import parse_text_grid, validate_sudoku, solve_sudoku, count_solutions

class TestSudoku(unittest.TestCase):

    def test_valid_sudoku(self):
        text = """
003000000
000008530
080040000
500000000
000090010
000600000
000000000
007005000
000000200
"""
        n, grid = parse_text_grid(text)
        ok, errs = validate_sudoku(n, grid)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

    def test_invalid_sudoku(self):
        text = """
113000000
000008530
080040000
500000000
000090010
000600000
000000000
007005000
000000200
"""
        n, grid = parse_text_grid(text)
        ok, errs = validate_sudoku(n, grid)
        self.assertFalse(ok)
        self.assertGreater(len(errs), 0)

    def test_solver(self):
        text = """
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
"""
        n, grid = parse_text_grid(text)
        solved, sol, cnt = solve_sudoku(n, grid)
        self.assertTrue(solved)
        self.assertIsNotNone(sol)

    def test_unique_check(self):
        text = """
003000000
000008530
080040000
500000000
000090010
000600000
000000000
007005000
000000200
"""
        n, grid = parse_text_grid(text)
        cnt = count_solutions(n, grid, limit=2)
        self.assertGreaterEqual(cnt, 1)

if __name__ == '__main__':
    unittest.main()
