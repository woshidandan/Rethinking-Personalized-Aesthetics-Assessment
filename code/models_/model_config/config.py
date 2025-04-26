VIT = {
    "input_resolution": 224,
    "patch_size": 14,
    "width": 1024,
    "layers": 24,
    "heads": 16,
    "output_dim": 768,
}

INPUT_DIMS = {
    "clip_input_num": 768,
    "mf_input_num": 256,
    "gt_input_num": 256,
    "people_answer_num": 5,
}

GROUPNORM = {
    "mf": {"num_channels": 256, "num_groups": 16},
    "pam": {"num_channels": 256, "num_groups": 16},
}

ADAPTER = {"reduction": 4}

FUSION = {"ratio": 0.7, "mode": "cat"}

DROPOUT_RATE = 0.7

HIDDEN_DIMS = {"layer1": 512, "layer2": 256}
