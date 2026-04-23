import subprocess


def test_train_hf_dataset_help_includes_text_transformer():
    result = subprocess.run(
        ["python3", "scripts/train/train_hf_dataset.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "text_transformer" in result.stdout
    assert "hf_text_encoder" in result.stdout
    assert "--d-model" in result.stdout
    assert "--nhead" in result.stdout
    assert "--text-num-layers" in result.stdout
    assert "--dim-feedforward" in result.stdout
    assert "--pretrained-text-model-name" in result.stdout
    assert "--save-trajectory" in result.stdout


def test_search_hf_text_dataset_help_includes_supported_schemes():
    result = subprocess.run(
        ["python3", "scripts/search/search_hf_text_dataset.py", "--help"],
        cwd="/home/hslyu/research/rework/GIF",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "gif" in result.stdout
    assert "tracin" in result.stdout
    assert "hyperinf" in result.stdout
    assert "second_influence" in result.stdout
    assert "freeze_influence" in result.stdout
