#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from motion_correction.motion_postprocess import correct_motion


class Joint:
    def __init__(self, name, parent, t_pose_translation, t_pose_rotation, retarget_tag=""):
        self.name = name
        self.parent = parent
        self.t_pose_translation = t_pose_translation
        self.t_pose_rotation = t_pose_rotation
        self.retarget_tag = retarget_tag


def create_test_rig():
    return [
        Joint("Hips", None, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], "Root"),
        Joint("Spine", "Hips", [0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 1.0]),
        Joint("LeftUpLeg", "Hips", [-0.1, -0.05, 0.0], [0.0, 0.0, 0.0, 1.0]),
        Joint("LeftLeg", "LeftUpLeg", [0.0, -0.4, 0.0], [0.0, 0.0, 0.0, 1.0]),
        Joint("LeftFoot", "LeftLeg", [0.0, -0.4, 0.0], [0.0, 0.0, 0.0, 1.0], "LeftFoot"),
        Joint("RightUpLeg", "Hips", [0.1, -0.05, 0.0], [0.0, 0.0, 0.0, 1.0]),
        Joint("RightLeg", "RightUpLeg", [0.0, -0.4, 0.0], [0.0, 0.0, 0.0, 1.0]),
        Joint("RightFoot", "RightLeg", [0.0, -0.4, 0.0], [0.0, 0.0, 0.0, 1.0], "RightFoot"),
        Joint("LeftArm", "Spine", [-0.3, 0.3, 0.0], [0.0, 0.0, 0.0, 1.0]),
        Joint("LeftHand", "LeftArm", [-0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], "LeftHand"),
        Joint("RightArm", "Spine", [0.3, 0.3, 0.0], [0.0, 0.0, 0.0, 1.0]),
        Joint("RightHand", "RightArm", [0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], "RightHand"),
    ]


if __name__ == "__main__":
    # Test data
    batch_size, num_frames, num_joints = 1, 60, 12

    hipTranslations = torch.randn(batch_size, num_frames, 3)
    jointRotations = torch.randn(batch_size, num_frames, num_joints, 4)
    jointRotations = jointRotations / jointRotations.norm(dim=-1, keepdim=True)

    contacts = torch.rand(batch_size, num_frames, 4)
    hipTranslationsInput = hipTranslations.clone()
    rotationsInput = jointRotations.clone()

    constraint_masks = {
        "Root": torch.zeros(num_frames),
        "FullBody": torch.zeros(num_frames),
        "LeftHand": torch.zeros(num_frames),
        "RightHand": torch.zeros(num_frames),
        "LeftFoot": torch.zeros(num_frames),
        "RightFoot": torch.zeros(num_frames),
    }

    working_rig = create_test_rig()

    # Run correction
    correct_motion(
        hipTranslations=hipTranslations,
        jointRotations=jointRotations,
        contacts=contacts,
        hipTranslationsInput=hipTranslationsInput,
        rotationsInput=rotationsInput,
        constraint_masks=constraint_masks,
        contact_threshold=0.5,
        root_margin=0.01,
        working_rig=working_rig,
    )

    print("Test completed successfully")
