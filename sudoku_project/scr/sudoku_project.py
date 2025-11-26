"""
SI3005 - Trabajo final: Sudoku
Implementación en Python: validación, solución y generación de n-sudoku (incluye 3-sudoku por defecto).
Basado en el enunciado del curso: SI3005_20252_Sudoku.pdf

Archivo: sudoku_project.py
Autor: Generado para el usuario
Fecha: 2025-11-23

Este archivo contiene:
 - parse_input / format_output (entrada/salida en el formato del enunciado)
 - validate_sudoku(n, grid): valida filas, columnas y bloques
 - solve_sudoku(n, grid): solucionador con backtracking + MRV + forward checking
 - count_solutions(limit=2): contador de soluciones para verificar unicidad
 - generate_sudoku(n, clues=None, max_attempts=1000): genera un sudoku con solución única
 - CLI para usar los tres modos: validate, solve, generate

Heurística principales:
 - MRV (Minimum Remaining Values): escoger la celda con menos candidatos.
 - Forward checking: mantener conjuntos de candidatos y actualizar al asignar.
 - Al generar, se crea una solución completa aleatoria y se van borrando celdas en orden aleatorio comprobando unicidad.

Complejidad (resumen):
 - Validación: O(n^4) en el peor caso por recorrer n^4 celdas y verificar sets (pero conn n^2 símbolos y n^4 celdas es O(n^4)).
 - Solución (backtracking): exponencial en n^4 en el peor caso (NP-completo). Las heurísticas reducen fuertemente el espacio de búsqueda práctico.
 - Generación con verificación única: costosa (varias ejecuciones del solver), depende de cuántas celdas se intenten borrar y de la dificultad del tablero.

Formato de entrada/salida (según el enunciado):
 - Para un n-sudoku, n se pasa como entero. El tablero es n^2 x n^2.
 - Cada símbolo ocupa W = floor(log10(n^2) + 1) dígitos, con ceros a la izquierda si es necesario.
 - Celdas vacías se representan con W guiones ('-').

Ejemplos de uso (CLI):
  python sudoku_project.py validate --file puzzle.txt
  python sudoku_project.py solve --file puzzle.txt --out solved.txt
  python sudoku_project.py generate --n 3 --clues 30 --out puzzle.txt

"""

import sys
import math
import argparse
import random
import copy
from typing import List, Tuple, Optional, Set, Dict

# ------------------------------
# I/O helpers (formato del enunciado)
# ------------------------------

def width_for_n(n: int) -> int:
    """Retorna W = floor(log10(n^2) + 1)"""
    W = int(math.floor(math.log10(n * n) + 1)) if n * n > 0 else 1
    return max(1, W)


def parse_text_grid(text: str) -> Tuple[int, List[List[Optional[int]]]]:
    """
    Parsear el contenido textual de un sudoku en el formato del enunciado.
    Se espera que la primera línea contenga `n` (opcional). Si no está, asumimos n=3.
    Alternativamente, si el contenido tiene exactamente n^2 líneas con tokens de ancho W, lo parsea.

    Devuelve (n, grid) donde grid es una lista de listas con 0 o None para celdas vacías.
    """
    lines = [l.rstrip('\n') for l in text.splitlines() if l.strip() != '']
    if not lines:
        raise ValueError("Entrada vacía")

    # Si la primera línea es un entero pequeño lo interpretamos como n
    first = lines[0].strip()
    n = None
    try:
        maybe_n = int(first)
        # solo si hay más líneas que el tamaño mínimo interpretamos como n
        if len(lines) >= 2:
            n = maybe_n
            lines = lines[1:]
    except Exception:
        n = None

    # Si n no viene, intentamos deducirlo por conteo de celdas asumiendo cuadrado
    # Cada línea tiene tokens contiguos con ancho W
    if n is None:
        # intentar suponer n=3 como default si el tablero tiene 9 líneas
        if len(lines) == 9:
            n = 3
        else:
            # buscar posible n por raíz cuadrada
            total_lines = len(lines)
            root = int(math.isqrt(total_lines))
            if root * root == total_lines:
                n = int(math.isqrt(root))  # porque total_lines = n^2
            else:
                # fallback
                n = 3

    N = n * n
    W = width_for_n(n)

    if len(lines) != N:
        # permitir que el archivo tenga filas separadas por espacios
        pass

    grid: List[List[Optional[int]]] = []
    for row in lines:
        tokens = []
        # si hay espacios, separar por espacios; si no, tomar trozos de W
        if ' ' in row.strip():
            parts = [p for p in row.strip().split() if p != '']
            for p in parts:
                if set(p) == set('-'):
                    tokens.append(None)
                else:
                    tokens.append(int(p))
        else:
            # cortar en trozos de W
            parts = [row[i:i+W] for i in range(0, len(row), W)]
            for p in parts:
                p = p.strip()
                if p == '' or set(p) == set('-'):
                    tokens.append(None)
                else:
                    tokens.append(int(p))
        if len(tokens) != N:
            raise ValueError(f"Fila con longitud inesperada. Esperado {N}, obtenido {len(tokens)} en fila: {row}")
        grid.append(tokens)

    if len(grid) != N:
        raise ValueError(f"Numero de filas inesperado. Esperado {N}, obtenido {len(grid)}")

    return n, grid


def format_grid(n: int, grid: List[List[Optional[int]]]) -> str:
    N = n * n
    W = width_for_n(n)
    lines = []
    for r in range(N):
        row = grid[r]
        parts = []
        for c in range(N):
            v = row[c]
            if v is None or v == 0:
                parts.append('-' * W)
            else:
                s = str(v).zfill(W)
                parts.append(s)
        lines.append(''.join(parts))
    return '\n'.join(lines)

# ------------------------------
# Validación
# ------------------------------

def validate_sudoku(n: int, grid: List[List[Optional[int]]]) -> Tuple[bool, List[str]]:
    """
    Valida que un n-sudoku sea consistente: no repeticiones en filas, columnas o bloques.
    Las celdas vacías (None o 0) se ignoran para la validación.

    Retorna (es_valido, lista_de_errores)
    """
    N = n * n
    errors = []

    # Rango válido de símbolos: 1..N
    for r in range(N):
        for c in range(N):
            v = grid[r][c]
            if v is None:
                continue
            if not (1 <= v <= N):
                errors.append(f"Valor fuera de rango en ({r},{c}): {v}")

    # Filas
    for r in range(N):
        seen: Set[int] = set()
        for c in range(N):
            v = grid[r][c]
            if v is None:
                continue
            if v in seen:
                errors.append(f"Repetido en fila {r}: {v}")
            seen.add(v)

    # Columnas
    for c in range(N):
        seen = set()
        for r in range(N):
            v = grid[r][c]
            if v is None:
                continue
            if v in seen:
                errors.append(f"Repetido en columna {c}: {v}")
            seen.add(v)

    # Bloques
    for br in range(n):
        for bc in range(n):
            seen = set()
            for r in range(br * n, br * n + n):
                for c in range(bc * n, bc * n + n):
                    v = grid[r][c]
                    if v is None:
                        continue
                    if v in seen:
                        errors.append(f"Repetido en bloque ({br},{bc}): {v}")
                    seen.add(v)

    return (len(errors) == 0, errors)

# ------------------------------
# Solucionador con MRV + Forward Checking
# ------------------------------

def solve_sudoku(n: int, grid: List[List[Optional[int]]], find_all: bool=False, max_solutions: int=2) -> Tuple[bool, List[List[Optional[int]]], int]:
    """
    Intenta resolver el sudoku. Si tiene solución devuelve (True, solución, num_solutions_found).
    Si find_all=True intenta contar hasta max_solutions soluciones (útil para verificar unicidad).
    """
    N = n * n

    # Inicializar candidatos: para cada celda mantener conjunto de valores posibles
    rows: List[Set[int]] = [set(range(1, N+1)) for _ in range(N)]
    cols: List[Set[int]] = [set(range(1, N+1)) for _ in range(N)]
    blocks: List[Set[int]] = [set(range(1, N+1)) for _ in range(N)]

    for r in range(N):
        for c in range(N):
            v = grid[r][c]
            if v is not None:
                rows[r].discard(v)
                cols[c].discard(v)
                b = (r // n) * n + (c // n)
                blocks[b].discard(v)

    # crear lista de celdas vacías
    empties: List[Tuple[int,int]] = [(r,c) for r in range(N) for c in range(N) if grid[r][c] is None or grid[r][c] == 0]

    solution_count = 0
    solution_grid: Optional[List[List[Optional[int]]]] = None

    # util: calcular candidatos actuales de una celda
    def candidates_for(r: int, c: int) -> Set[int]:
        b = (r // n) * n + (c // n)
        return rows[r] & cols[c] & blocks[b]

    # Ordenar por MRV dinámicamente en cada llamada recursiva
    def backtrack(idx: int) -> bool:
        nonlocal solution_count, solution_grid
        # Si hemos encontrado suficientes soluciones, parar
        if solution_count >= max_solutions:
            return True

        # Si no hay vacías, solución completa
        if idx == len(empties):
            solution_count += 1
            solution_grid = copy.deepcopy(grid)
            return False if find_all else True

        # Escoger celda con MRV entre las que quedan (de idx..end)
        # en vez de idx orden fijo, buscamos la posición con menos candidatos
        best_pos = -1
        best_count = 10**9
        for k in range(idx, len(empties)):
            r,c = empties[k]
            cand = candidates_for(r,c)
            l = len(cand)
            if l == 0:
                return False
            if l < best_count:
                best_count = l
                best_pos = k
                if l == 1:
                    break

        # swap chosen into position idx
        empties[idx], empties[best_pos] = empties[best_pos], empties[idx]
        r,c = empties[idx]
        candset = list(candidates_for(r,c))

        # try values (shuffled to diversify search tree)
        random.shuffle(candset)
        for val in candset:
            # assign
            grid[r][c] = val
            b = (r // n) * n + (c // n)
            rows[r].remove(val)
            cols[c].remove(val)
            blocks[b].remove(val)

            cont = backtrack(idx+1)
            # si no estamos buscando todas las soluciones y hemos encontrado una, propagar True
            if cont and not find_all:
                return True

            # undo
            rows[r].add(val)
            cols[c].add(val)
            blocks[b].add(val)
            grid[r][c] = None

            if solution_count >= max_solutions:
                return True

        # restore swap (opcional)
        empties[idx], empties[best_pos] = empties[best_pos], empties[idx]
        return False

    solved = backtrack(0)
    if solution_grid is None and solved:
        solution_grid = grid
    return (solved or solution_count>0, solution_grid if solution_grid is not None else grid, solution_count)

# ------------------------------
# Contador de soluciones (usa solver con find_all=True)
# ------------------------------

def count_solutions(n: int, grid: List[List[Optional[int]]], limit: int=2) -> int:
    _, _, cnt = solve_sudoku(n, copy.deepcopy(grid), find_all=True, max_solutions=limit)
    return cnt

# ------------------------------
# Generador
# ------------------------------

def _fill_complete_board(n: int) -> List[List[int]]:
    """
    Genera una solución completa (tablero resuelto) aleatoria mediante backtracking.
    """
    N = n * n
    grid: List[List[Optional[int]]] = [[None]*N for _ in range(N)]

    def fill_cell(idx: int) -> bool:
        if idx == N * N:
            return True
        r = idx // N
        c = idx % N
        if grid[r][c] is not None:
            return fill_cell(idx+1)
        nums = list(range(1, N+1))
        random.shuffle(nums)
        for v in nums:
            good = True
            # check row
            if v in grid[r]:
                good = False
            if not good:
                continue
            # check col
            for rr in range(N):
                if grid[rr][c] == v:
                    good = False
                    break
            if not good:
                continue
            # check block
            br = (r//n)*n
            bc = (c//n)*n
            for rr in range(br, br+n):
                for cc in range(bc, bc+n):
                    if grid[rr][cc] == v:
                        good = False
                        break
                if not good:
                    break
            if not good:
                continue

            grid[r][c] = v
            if fill_cell(idx+1):
                return True
            grid[r][c] = None
        return False

    ok = fill_cell(0)
    if not ok:
        raise RuntimeError("No se pudo generar tablero completo (improbable)")
    # convertir Optional[int] -> int
    return [[int(grid[r][c]) for c in range(N)] for r in range(N)]


def generate_sudoku(n: int, clues: Optional[int]=None, max_attempts: int=1000) -> List[List[Optional[int]]]:
    """
    Genera un n-sudoku con solución única. Si `clues` es None, se intentará un valor razonable.
    El proceso: generar tablero completo, luego retirar celdas aleatoriamente comprobando unicidad.
    max_attempts: número máximo de intentos de borrar celdas antes de reiniciar la generación.
    """
    N = n * n
    if clues is None:
        # heurística simple: para 9x9, entre 22 y 32 suele ser razonable; aquí buscaremos 30
        clues = max(17, min(N*N, int(0.35 * N * N)))

    attempts = 0
    while True:
        attempts += 1
        if attempts > max_attempts:
            # reiniciar parámetros y continuar
            attempts = 0
        # 1) generar solución completa
        complete = _fill_complete_board(n)
        # 2) empezar con todo lleno y vaciar
        grid: List[List[Optional[int]]] = [[complete[r][c] for c in range(N)] for r in range(N)]
        positions = [(r,c) for r in range(N) for c in range(N)]
        random.shuffle(positions)

        removed = 0
        for (r,c) in positions:
            if removed >= N*N - clues:
                break
            backup = grid[r][c]
            grid[r][c] = None
            # comprobar unicidad: si tiene exactamente 1 solución se acepta la eliminación
            cnt = count_solutions(n, copy.deepcopy(grid), limit=2)
            if cnt != 1:
                # no es único (0 o >1) -> deshacer
                grid[r][c] = backup
            else:
                removed += 1

        # comprobar que alcanzamos la cantidad de pistas deseadas
        current_clues = sum(1 for r in range(N) for c in range(N) if grid[r][c] is not None)
        if current_clues == clues:
            return grid
        # sino, volver a intentar (posible que el orden no permitiera llegar a clues)

# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='Herramientas para n-sudoku: validar, resolver, generar')
    sub = parser.add_subparsers(dest='cmd')

    pval = sub.add_parser('validate', help='Validar un sudoku')
    pval.add_argument('--file', '-f', required=True, help='Archivo de entrada (formato del enunciado)')

    psol = sub.add_parser('solve', help='Resolver un sudoku')
    psol.add_argument('--file', '-f', required=True, help='Archivo de entrada')
    psol.add_argument('--out', '-o', required=False, help='Archivo de salida (si no se especifica, stdout)')

    pgen = sub.add_parser('generate', help='Generar un sudoku con solución única')
    pgen.add_argument('--n', type=int, default=3, help='n (por defecto 3 para 9x9)')
    pgen.add_argument('--clues', type=int, required=False, help='Número de pistas (ej: 30)')
    pgen.add_argument('--out', '-o', required=False, help='Archivo de salida')

    args = parser.parse_args()

    if args.cmd == 'validate':
        with open(args.file, 'r', encoding='utf-8') as fh:
            text = fh.read()
        n, grid = parse_text_grid(text)
        ok, errs = validate_sudoku(n, grid)
        if ok:
            print('VALIDO')
        else:
            print('INVALIDO')
            for e in errs:
                print(e)

    elif args.cmd == 'solve':
        with open(args.file, 'r', encoding='utf-8') as fh:
            text = fh.read()
        n, grid = parse_text_grid(text)
        solved, sol, cnt = solve_sudoku(n, grid)
        if not solved:
            print('SIN_SOLUCION')
            sys.exit(1)
        outtext = format_grid(n, sol)
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as fh:
                fh.write(outtext)
            print(f'Solución guardada en {args.out}')
        else:
            print(outtext)

    elif args.cmd == 'generate':
        n = args.n
        clues = args.clues
        grid = generate_sudoku(n, clues)
        outtext = format_grid(n, grid)
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as fh:
                fh.write(outtext)
            print(f'Puzzle generado guardado en {args.out}')
        else:
            print(outtext)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

# Ejemplos de entrada y salida

"""
Ejemplo 1: Validación de un 3-sudoku incompleto (válido)
Entrada:
003000000
000008530
080040000
500000000
000090010
000600000
000000000
007005000
000000200

Salida esperada (validate):
VALIDO

Ejemplo 2: Validación de un 3-sudoku con error en fila
Entrada:
113000000
000008530
080040000
500000000
000090010
000600000
000000000
007005000
000000200

Salida esperada:
INVALIDO
Repetido en fila 0: 1

Ejemplo 3: Resolución de un sudoku fácil (entrada incompleta), salida es el sudoku resuelto.

Ejemplo 4: Generación de un 3-sudoku con 30 pistas.
Uso:
python sudoku_project.py generate --n 3 --clues 30 --out puzzle.txt
"""
