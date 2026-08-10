"""多周期 Colab notebook 的结构和安全边界测试。"""

from pathlib import Path

import nbformat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "nextday_multi_horizon_validation_colab.ipynb"


def test_multi_horizon_notebook_is_clean_and_validation_only() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in code_cells)

    assert "## Goal" in markdown
    assert "## Setup" in markdown
    assert "## Steps" in markdown
    assert "## Checks" in markdown
    assert "## Next Steps" in markdown
    assert "evaluate_validation_horizons" in code
    assert "evaluate_test=False" in code
    assert 'result["test_status"] == "locked_not_accessed"' in code
    assert "evaluate_best_checkpoints" not in code
    assert 'split="test"' not in code
    assert "train(" not in code
    assert all(cell.execution_count is None and not cell.outputs for cell in code_cells)

    for index, cell in enumerate(code_cells):
        compile(cell.source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")
