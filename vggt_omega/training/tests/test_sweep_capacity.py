def test_sweep_capacity_parses():
    from vggt_omega.training import sweep_capacity

    args = sweep_capacity.parse_args(["--caps", "24,32", "--img-size", "512"])
    assert args.caps == [24, 32]
