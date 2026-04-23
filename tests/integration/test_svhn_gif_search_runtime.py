import subprocess


def test_search_gif_svhn_runtime_help_includes_grid_arguments():
    result = subprocess.run(
        ["python3", "scripts/experiments/influence/search_gif_svhn_runtime.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--tol-values" in result.stdout
    assert "--num-target-batches-values" in result.stdout
    assert "--param-ratio-values" in result.stdout
    assert "--max-iter-values" in result.stdout
    assert "--mu-values" in result.stdout
    assert "--edit-scale-values" in result.stdout
    assert "--solver-power-iters-values" in result.stdout
    assert "--gif-max-self-acc-values" in result.stdout
    assert "--limit" in result.stdout
    assert "--dry-run" in result.stdout
