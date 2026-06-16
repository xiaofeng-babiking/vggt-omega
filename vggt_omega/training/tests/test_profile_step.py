def test_profile_step_module_imports_and_parses():
    from vggt_omega.training import profile_step
    args = profile_step.parse_args(["--config", "vggt_omega/training/config/train_smoke.yaml",
                                    "--warm", "2", "--steps", "5"])
    assert args.warm == 2 and args.steps == 5 and args.config.endswith("train_smoke.yaml")
