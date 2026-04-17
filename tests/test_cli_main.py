from nlos_cs.cli.main import app, build_parser


def test_build_parser_returns_argparse_parser():
    parser = build_parser()
    assert parser.prog == "nlos-cs"


def test_app_with_no_args_prints_help_and_returns_zero(capsys):
    code = app([])
    captured = capsys.readouterr()

    assert code == 0
    assert "usage:" in captured.out.lower()
    assert "nlos-cs" in captured.out


def test_app_info_command(capsys):
    code = app(["info"])
    captured = capsys.readouterr()

    assert code == 0
    assert "nlos-cs version:" in captured.out
    assert "Status: early rebuild" in captured.out


def test_app_build_operator_placeholder(capsys):
    code = app(["build-operator"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Command 'build-operator' is not wired yet." in captured.out


def test_app_reconstruct_placeholder(capsys):
    code = app(["reconstruct"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Command 'reconstruct' is not wired yet." in captured.out


def test_app_discrim_placeholder(capsys):
    code = app(["discrim"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Command 'discrim' is not wired yet." in captured.out


def test_app_robustness_placeholder(capsys):
    code = app(["robustness"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Command 'robustness' is not wired yet." in captured.out