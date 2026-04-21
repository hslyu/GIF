import subprocess


def test_search_mnist_model_help_includes_all_extended_schemes():
    result = subprocess.run(
        ["python3", "scripts/search/search_mnist_model.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "gif" in result.stdout
    assert "tracin" in result.stdout
    assert "influence" in result.stdout
    assert "second_influence" in result.stdout
    assert "freeze_influence" in result.stdout
    assert "hyperinf" in result.stdout
    assert "lissa" in result.stdout
    assert "cg" in result.stdout
    assert "datainf" in result.stdout
    assert "highest_k_gradients" not in result.stdout
    assert "fcn_lora" in result.stdout
    assert "--trajectory-dir" in result.stdout
    assert "--hyperinf-beta-scale" in result.stdout
    assert "--datainf-damping" in result.stdout
    assert "--solver-damping" in result.stdout


def test_run_gif_unlearning_help_includes_tracin_hyperinf_and_datainf():
    result = subprocess.run(
        ["python3", "scripts/search/run_gif_unlearning_mnist.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "gif" in result.stdout
    assert "tracin" in result.stdout
    assert "influence" in result.stdout
    assert "second_influence" in result.stdout
    assert "freeze_influence" in result.stdout
    assert "hyperinf" in result.stdout
    assert "lissa" in result.stdout
    assert "cg" in result.stdout
    assert "datainf" in result.stdout
    assert "highest_k_gradients" not in result.stdout
    assert "--trajectory-dir" in result.stdout
    assert "--hyperinf-beta-scale" in result.stdout
    assert "--datainf-damping" in result.stdout
    assert "--solver-damping" in result.stdout


def test_search_gif_unlearning_help_includes_tracin_hyperinf_and_datainf():
    result = subprocess.run(
        ["python3", "scripts/search/search_gif_unlearning_mnist.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "gif" in result.stdout
    assert "tracin" in result.stdout
    assert "influence" in result.stdout
    assert "second_influence" in result.stdout
    assert "freeze_influence" in result.stdout
    assert "hyperinf" in result.stdout
    assert "lissa" in result.stdout
    assert "cg" in result.stdout
    assert "datainf" in result.stdout
    assert "highest_k_gradients" not in result.stdout
    assert "--trajectory-dir" in result.stdout
    assert "--hyperinf-beta-scale" in result.stdout
    assert "--datainf-damping" in result.stdout
    assert "--solver-damping" in result.stdout
