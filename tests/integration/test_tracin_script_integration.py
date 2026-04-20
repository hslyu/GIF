import subprocess


def test_search_mnist_model_help_includes_tracin():
    result = subprocess.run(
        ["python3", "scripts/search/search_mnist_model.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tracin" in result.stdout
    assert "--trajectory-dir" in result.stdout


def test_run_gif_unlearning_help_includes_tracin():
    result = subprocess.run(
        ["python3", "scripts/search/run_gif_unlearning_mnist.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tracin" in result.stdout
    assert "--trajectory-dir" in result.stdout


def test_search_gif_unlearning_help_includes_tracin():
    result = subprocess.run(
        ["python3", "scripts/search/search_gif_unlearning_mnist.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tracin" in result.stdout
    assert "--trajectory-dir" in result.stdout
