"""
tests/test_notebooks.py – static checks for all five notebooks.

Checks performed:
1. nbformat validity (nbformat.validate).
2. No stored outputs in any cell.
3. A static name-binding check over every code cell (stdlib AST):
   - Lines starting with ! or % (including backslash-continued) are skipped.
   - Cells starting with %% are skipped entirely.
   - Any name used before it is bound in the accumulated scope raises an error.
   The check catches a deliberately broken synthetic notebook (see test at bottom).
4. An AST check that notebooks 02-04 contain no string constant "lesion" or
   integer 42 outside the first cell.

All checks run without executing any notebook code.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

try:
    import nbformat
    try:
        from nbformat.validate import validate as nb_validate, ValidationError
    except ImportError:
        # nbformat >= 5.10 moved the public API
        from nbformat.validator import validate as nb_validate  # type: ignore[no-redef]
        ValidationError = Exception  # type: ignore[assignment,misc]
    _HAS_NBFORMAT = True
except ImportError:
    _HAS_NBFORMAT = False

pytestmark = pytest.mark.skipif(
    not _HAS_NBFORMAT,
    reason="nbformat not installed",
)

_NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
_NOTEBOOKS = sorted(_NB_DIR.glob("*.ipynb"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_nb(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def _code_cells(nb) -> list[dict]:
    return [c for c in nb.cells if c.cell_type == "code"]


def _strip_magic_lines(source: str) -> str:
    """Remove lines that start with !, %, or are continuations of such lines."""
    out_lines = []
    skip_continuation = False
    for raw_line in source.splitlines():
        stripped = raw_line.lstrip()
        if skip_continuation:
            # Previous line ended with backslash — still part of magic/shell.
            skip_continuation = raw_line.rstrip().endswith("\\")
            continue
        if stripped.startswith("!") or stripped.startswith("%"):
            skip_continuation = raw_line.rstrip().endswith("\\")
            continue
        out_lines.append(raw_line)
    return "\n".join(out_lines)


def _is_magic_cell(source: str) -> bool:
    """Return True if the cell starts with %% (cell magic)."""
    return source.lstrip().startswith("%%")


# ---------------------------------------------------------------------------
# Static name-binding checker
# ---------------------------------------------------------------------------

class _NameBindingChecker(ast.NodeVisitor):
    """Track names bound and used in an accumulated set of AST statements.

    Bound = assigned, imported, defined (def/class), loop targets, etc.
    Used  = loaded names.

    We also treat a small set of names as always-available (builtins +
    common Colab/Jupyter globals).
    """

    # Names that are always in scope (builtins + Colab-injected globals).
    _ALWAYS_BOUND: frozenset[str] = frozenset({
        # Python builtins
        "abs", "all", "any", "ascii", "bin", "bool", "breakpoint",
        "bytearray", "bytes", "callable", "chr", "classmethod", "compile",
        "complex", "copyright", "credits", "delattr", "dict", "dir", "divmod",
        "enumerate", "eval", "exec", "exit", "filter", "float", "format",
        "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex",
        "id", "input", "int", "isinstance", "issubclass", "iter", "len",
        "license", "list", "locals", "map", "max", "memoryview", "min",
        "next", "object", "oct", "open", "ord", "pow", "print", "property",
        "quit", "range", "repr", "reversed", "round", "set", "setattr",
        "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple",
        "type", "vars", "zip",
        # Common dunders
        "__name__", "__file__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "__annotations__",
        # Jupyter/Colab injected
        "In", "Out", "get_ipython", "_", "__",
        # Exception variable in except clauses is handled specially
        "Exception", "ValueError", "KeyError", "RuntimeError",
        "ImportError", "OSError", "IOError", "FileNotFoundError",
        "NotImplementedError", "AttributeError", "TypeError", "IndexError",
        "StopIteration", "KeyboardInterrupt", "SystemExit", "True", "False",
        "None",
        # Common type aliases used in annotations
        "Optional", "List", "Dict", "Tuple", "Union", "Any",
    })

    def __init__(self) -> None:
        self.bound: set[str] = set(self._ALWAYS_BOUND)
        self.errors: list[str] = []

    def _bind(self, name: str) -> None:
        self.bound.add(name)

    def _check_use(self, name: str, lineno: int) -> None:
        if name not in self.bound:
            self.errors.append(
                f"line {lineno}: name '{name}' used before it is bound"
            )

    # --- binding visitors ---

    def visit_Assign(self, node: ast.Assign) -> None:
        # Visit the value first (RHS), then bind targets.
        self.visit(node.value)
        for target in node.targets:
            self._bind_target(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self.visit(node.value)
        self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind(node.target.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            self._bind(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                return  # star import: we can't track; allow anything
            name = alias.asname if alias.asname else alias.name
            self._bind(name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name)
        # Don't recurse into the body — inner scopes have their own bindings.

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    # Comprehensions bind their loop variables into the enclosing scope
    # (in Python 3 generator expressions have their own scope, but list/dict/set
    # comprehensions' targets leak in Python 2; in Python 3 they don't leak, but
    # for the purpose of this static check we allow them so that [i for i in x]
    # doesn't produce spurious errors on the 'i' reference inside the comprehension).
    def _visit_comprehension(self, node) -> None:
        for generator in node.generators:
            self.visit(generator.iter)
            self._bind_target(generator.target)
            for cond in generator.ifs:
                self.visit(cond)
        # Visit the element expression(s).
        if hasattr(node, "elt"):
            self.visit(node.elt)
        elif hasattr(node, "key"):
            self.visit(node.key)
            self.visit(node.value)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._bind_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type:
            self.visit(node.type)
        if node.name:
            self._bind(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._bind(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        pass  # affects enclosing scope, not our flat model

    # --- use visitors ---

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._check_use(node.id, node.lineno)

    # --- helpers ---

    def _bind_target(self, node) -> None:
        if isinstance(node, ast.Name):
            self._bind(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._bind_target(elt)
        elif isinstance(node, ast.Starred):
            self._bind_target(node.value)
        elif isinstance(node, ast.Attribute):
            pass  # attribute assignment doesn't create a new name


def check_name_binding(code_cells: list[str]) -> list[str]:
    """Run the checker over all code cells in order; return list of error strings."""
    checker = _NameBindingChecker()
    all_errors: list[str] = []

    for cell_idx, source in enumerate(code_cells):
        if _is_magic_cell(source):
            continue
        cleaned = _strip_magic_lines(source)
        if not cleaned.strip():
            continue
        try:
            tree = ast.parse(cleaned)
        except SyntaxError as exc:
            all_errors.append(f"cell {cell_idx}: SyntaxError: {exc}")
            continue

        checker.errors = []
        checker.visit(tree)

        # Bind all names assigned at top level in this cell for future cells.
        # (checker.bound is mutated in place)

        if checker.errors:
            for err in checker.errors:
                all_errors.append(f"cell {cell_idx}: {err}")

    return all_errors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=[p.name for p in _NOTEBOOKS])
def test_nbformat_valid(nb_path: Path):
    nb = _load_nb(nb_path)
    nb_validate(nb)  # raises ValidationError on failure


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=[p.name for p in _NOTEBOOKS])
def test_no_stored_outputs(nb_path: Path):
    nb = _load_nb(nb_path)
    for i, cell in enumerate(_code_cells(nb)):
        assert cell.outputs == [], (
            f"{nb_path.name} cell {i} has stored outputs; clear all outputs before committing."
        )
        assert cell.execution_count is None, (
            f"{nb_path.name} cell {i} has a non-null execution_count; clear before committing."
        )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=[p.name for p in _NOTEBOOKS])
def test_name_binding(nb_path: Path):
    nb = _load_nb(nb_path)
    sources = [c.source for c in _code_cells(nb)]
    errors = check_name_binding(sources)
    assert errors == [], (
        f"{nb_path.name} has unbound name references:\n" + "\n".join(errors)
    )


@pytest.mark.parametrize("nb_path", [
    p for p in _NOTEBOOKS
    if p.name.startswith("02_") or p.name.startswith("03_") or p.name.startswith("04_")
], ids=lambda p: p.name)
def test_no_hardcoded_lesion_or_42_outside_first_cell(nb_path: Path):
    """Notebooks 02-04 must not embed string 'lesion' or int 42 outside cell 0.

    These values come from the SPLIT_MODE and SEED parameters (cell 0).
    Hardcoding them in later cells would make the notebook silently ignore
    the parameter cell.
    """
    nb = _load_nb(nb_path)
    cells = _code_cells(nb)
    if len(cells) <= 1:
        return  # nothing to check

    errors: list[str] = []
    for cell_idx, cell in enumerate(cells[1:], start=1):
        if _is_magic_cell(cell.source):
            continue
        cleaned = _strip_magic_lines(cell.source)
        if not cleaned.strip():
            continue
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str) and node.value == "lesion":
                    errors.append(
                        f"cell {cell_idx} line {node.lineno}: "
                        "string constant 'lesion' (use SPLIT_MODE)"
                    )
                if isinstance(node.value, int) and node.value == 42:
                    errors.append(
                        f"cell {cell_idx} line {node.lineno}: "
                        "integer constant 42 (use SEED)"
                    )

    assert errors == [], (
        f"{nb_path.name} has hard-coded values outside the parameter cell:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Demonstrate the checker catching a broken synthetic notebook
# ---------------------------------------------------------------------------

def _make_broken_nb_source() -> dict:
    """Return a notebook dict with an undefined name used in cell 1."""
    cell0 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "x = 1\n",
    }
    cell1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "print(undefined_name)\n",  # 'undefined_name' never bound
    }
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [cell0, cell1],
    }


def test_checker_catches_broken_notebook():
    """The name-binding checker must detect undefined_name in the broken notebook."""
    nb = _make_broken_nb_source()
    sources = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    errors = check_name_binding(sources)
    assert any("undefined_name" in e for e in errors), (
        "Expected checker to flag 'undefined_name', but errors were: " + str(errors)
    )


def test_checker_passes_on_valid_notebook():
    """A notebook where all names are defined before use should produce no errors."""
    cell0 = "import os\nx = 1\n"
    cell1 = "y = x + 1\nprint(os.getcwd())\n"
    errors = check_name_binding([cell0, cell1])
    assert errors == []
