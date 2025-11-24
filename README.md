# Proyecto Final SI3005 – Sudoku (Python)

Este repositorio contiene la implementación completa del proyecto de Sudoku solicitado en el curso **SI3005 – Análisis y Diseño de Algoritmos**, incluyendo:

* Validación de un **n‑sudoku**.
* Resolución de un **n‑sudoku** mediante *backtracking*, MRV y forward checking.
* Generación de un **n‑sudoku con solución única**.
* Herramientas de línea de comando (CLI).

El proyecto está implementado en **Python 3**.

---

## 📁 Estructura del repositorio

```
📦 sudoku-project
 ┣ 📂 src
 ┃ ┗ sudoku_project.py
 ┣ 📂 examples
 ┃ ┣ example_valid.txt
 ┃ ┣ example_invalid.txt
 ┃ ┗ example_generated.txt
 ┣ 📂 tests
 ┃ ┗ test_sudoku.py
 ┣ README.md
 ┗ requirements.txt
```

---

## ▶ Requisitos

```
Python >= 3.9
```

No se usan librerías externas.

---

## ▶ Uso (CLI)

### **1. Validar un sudoku**

```
python sudoku_project.py validate --file puzzle.txt
```

### **2. Resolver un sudoku**

```
python sudoku_project.py solve --file puzzle.txt --out solucion.txt
```

### **3. Generar un sudoku con solución única**

```
python sudoku_project.py generate --n 3 --clues 30 --out puzzle.txt
```

---

## 🧪 Ejemplos incluidos

Los archivos dentro de `examples/` muestran:

* Un sudoku válido.
* Un sudoku inválido.
* Un sudoku generado automáticamente.

---

## 🧪 Tests unitarios

El archivo `tests/test_sudoku.py` permite verificar:

* Validación correcta.
* Solución correcta.
* Generación con unicidad.

Para ejecutarlos:

```
python -m unittest tests/test_sudoku.py
```

---

## 👨‍💻 Autor

Generado por Juan Miguel Londoño Castrillon.
