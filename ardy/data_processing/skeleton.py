# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ardy.assets import skeleton_asset_path
from ardy.data_processing.bvh import Bvh

_DEFAULT_G1_XML = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))


class Bone:
    """Abstraction for a bone in character skeleton."""

    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []  # bvh only
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []

        # asf specific
        self.dir = np.zeros(3)
        self.len = 0
        # bvh specific
        self.offset = np.zeros(3)  # default offset for position
        self.offset_rot = None  # rotation for custom nv bvh

        # inferred info
        self.pos = np.zeros(3)
        self.end = np.zeros(3)

    def __repr__(self):
        return f"{self.name}"


class Skeleton:
    """Abstraction for character skeleton."""

    def __init__(self):
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.dof_name = ["x", "y", "z"]
        self.root = None

    def get_bones_names(self):
        return [x.name for x in self.bones]

    def get_parent_indices(self):
        parent_indices = [-1] * len(self.bones)
        for bone in self.bones:
            if bone.parent:
                parent_indices[bone.id] = bone.parent.id
        return parent_indices

    def get_neutral_joints(self):
        joints = []
        for bone in self.bones:
            joints.append(bone.pos)
        joints = np.stack(joints, axis=0)
        return joints

    def load_from_bvh(self, fname, exclude_bones=None, spec_channels=None):
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()
        with open(fname) as f:
            mocap = Bvh(f.read())

        joint_names = list(
            filter(
                lambda x: all([t not in x for t in exclude_bones]),
                mocap.get_joints_names(),
            )
        )
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = 1.0
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = mocap.joint_channels(self.root.name)
        self.root.offset = np.array(mocap.joint_offset(self.root.name)) * self.len_scale
        self.root.offset_rot = mocap.joint_offset_rot(self.root.name)
        if self.root.offset_rot is not None:
            self.root.offset_rot = np.array(self.root.offset_rot)
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint
            bone.channels = spec_channels[joint] if joint in spec_channels.keys() else mocap.joint_channels(joint)
            bone.dof_index = [dof_ind[x[0].lower()] for x in bone.channels]
            bone.offset = np.array(mocap.joint_offset(joint)) * self.len_scale
            bone.offset_rot = mocap.joint_offset_rot(joint)
            if bone.offset_rot is not None:
                bone.offset_rot = np.array(bone.offset_rot)
            bone.lb = [-180.0] * 3
            bone.ub = [180.0] * 3
            self.bones.append(bone)
            self.name2bone[joint] = bone

        # for bone in self.bones:
        # print(bone.name, bone.channels, bone.offset)

        for bone in self.bones[1:]:
            parent_name = mocap.joint_parent(bone.name).name
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                child_vals = [str(node) for node in mocap.get_joint(bone.name).children]
                if "End Site" in child_vals:
                    end_site_idx = child_vals.index("End Site")
                    end_site_offset = mocap.get_joint(bone.name).children[end_site_idx]["OFFSET"]
                    bone.end = bone.pos + np.array([float(x) for x in end_site_offset]) * self.len_scale
                else:
                    pass
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def load_from_xml(self, fname, exclude_bones=None, spec_channels=None):
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()

        import xml.etree.ElementTree as ET

        tree = ET.parse(fname)
        root = tree.getroot()

        def map_bone_name(body_name):
            if body_name == "torso_link":
                return "waist_pitch_skel"
            elif "_link" in body_name:
                return body_name.replace("_link", "_skel")
            else:
                return body_name + "_skel"

        bodies = root.find("worldbody").findall(".//body")
        name_to_body = {map_bone_name(body.get("name")): body for body in bodies}
        joint_names = [body.get("name") for body in bodies]

        R_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
        x_forward_to_y_forward = Rotation.from_euler("z", -90, degrees=True)
        mujoco_to_ardy = R_zup_to_yup * x_forward_to_y_forward

        def mujoco_joint_translation(body_name):
            body = name_to_body[body_name]
            if "pos" in body.attrib:
                result = np.array([100 * float(x) for x in body.get("pos").strip().split(" ")])
                return mujoco_to_ardy.apply(result)
            else:
                return np.zeros(3)

        parent_map = {
            map_bone_name(child.get("name")): map_bone_name(parent.get("name"))
            for parent in root.iter()
            if parent.get("name")
            for child in parent
            if child.get("name")
        }

        self.len_scale = 1.0
        self.root = Bone()
        self.root.id = 0
        self.root.name = map_bone_name(joint_names[0])
        self.root.offset = mujoco_joint_translation(self.root.name)
        self.root.offset_rot = np.eye(3)

        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = map_bone_name(joint)
            bone.offset = mujoco_joint_translation(bone.name)
            bone.offset_rot = np.eye(3)
            self.bones.append(bone)
            self.name2bone[bone.name] = bone

        for bone in self.bones[1:]:
            parent_name = parent_map[bone.name]
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)

    def forward_bvh(self, bone):
        if bone.parent:
            bone.pos = bone.parent.pos + bone.offset
        else:
            bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)


class SkeletonG1(Skeleton):
    """Add functionality specific to the G1 including loading from XML specific to G1 and adding
    dummy joints."""

    def load_from_xml(self, fname, exclude_bones=None, spec_channels=None):
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()

        import xml.etree.ElementTree as ET

        tree = ET.parse(fname)
        root = tree.getroot()

        def map_bone_name(body_name):
            if body_name == "torso_link":
                return "waist_pitch_skel"
            elif "_link" in body_name:
                return body_name.replace("_link", "_skel")
            else:
                return body_name + "_skel"

        bodies = root.find("worldbody").findall(".//body")
        name_to_body = {map_bone_name(body.get("name")): body for body in bodies}
        joint_names = [body.get("name") for body in bodies]

        R_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
        x_forward_to_y_forward = Rotation.from_euler("z", -90, degrees=True)
        mujoco_to_ardy = R_zup_to_yup * x_forward_to_y_forward

        def mujoco_joint_translation(body_name):
            body = name_to_body[body_name]
            if "pos" in body.attrib:
                result = np.array([100 * float(x) for x in body.get("pos").strip().split(" ")])
                return mujoco_to_ardy.apply(result)
            else:
                return np.zeros(3)

        parent_map = {
            map_bone_name(child.get("name")): map_bone_name(parent.get("name"))
            for parent in root.iter()
            if parent.get("name")
            for child in parent
            if child.get("name")
        }

        self.len_scale = 1.0
        self.root = Bone()
        self.root.id = 0
        self.root.name = map_bone_name(joint_names[0])
        self.root.offset = mujoco_joint_translation(self.root.name)
        self.root.offset_rot = np.eye(3)

        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = map_bone_name(joint)
            bone.offset = mujoco_joint_translation(bone.name)
            bone.offset_rot = np.eye(3)
            self.bones.append(bone)
            self.name2bone[bone.name] = bone

        for bone in self.bones[1:]:
            parent_name = parent_map[bone.name]
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)

    def add_joint(self, parent_name, bone_name, offset):
        new_bone = Bone()
        new_bone.name = bone_name
        new_bone.parent = next((x for x in self.bones if x.name == parent_name))
        new_bone.parent.child.append(new_bone)
        new_bone.offset = offset
        new_bone.offset_rot = np.eye(3)

        parent_index = next((i for i, x in enumerate(self.bones) if x.name == parent_name))
        self.bones.insert(parent_index + 1, new_bone)
        self.name2bone[bone_name] = new_bone


def load_bvh_animation(
    fname: str,
    skeleton: Skeleton,
    rot_order: Optional[str] = "native",
    backend: Optional[str] = "np",
    return_quat: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a bvh file for one motion sequence, and Skeleton object containing the neutral
    skeleton, return root translation and joint rotation matrics for the motion sequence.

    Args:
        fname (str): full path of the bvh file
        skeleton (Skeleton): Skeleton object initialized by the bvh file, containing the neutral skeleton
        rot_order (str): one of "native", "ZXY", "YXZ", ....
        return_quat (bool): if True, return joint rotation quaternions instead of rotation matrices
    Returns:
        torch.Tensor: [T, 3] root translations
        torch.Tensor: [T, J, 3, 3] joint rotation matrix
    """
    with open(fname) as f:
        mocap = Bvh(f.read(), backend=backend)

    # assume all joints are same ordering, load in with native ordering
    root_channels = mocap.joint_channels(skeleton.root.name)
    pos_channels = [channel for channel in root_channels if channel.endswith("position")]
    rot_channels = [channel for channel in root_channels if channel.endswith("rotation")]

    root_trans = np.array(mocap.frames_joint_channels(skeleton.root.name, pos_channels))

    if backend == "np":
        # NOTE: assumes rot channel ordering is the same for all joints
        joint_eulers = mocap.frames_joints_channels(skeleton.get_bones_names(), rot_channels)
        joint_eulers = np.deg2rad(joint_eulers)
    elif backend == "graph":
        joint_eulers = []
        for bone in skeleton.bones:
            bone_channels = mocap.joint_channels(bone.name)
            bone_rot_channels = [channel for channel in bone_channels if channel.endswith("rotation")]
            assert bone_rot_channels == rot_channels, "Rotation channel ordering is not consistent across joints!"
            # use native rotation order
            euler = np.deg2rad(np.array(mocap.frames_joint_channels(bone.name, rot_channels)))
            joint_eulers.append(euler)
        joint_eulers = np.stack(joint_eulers, axis=1)
    else:
        raise ValueError(f"Unknown backend for BVH loading: {backend}")

    if rot_order == "native":
        rot_order = ""
        for axis in rot_channels:
            rot_order += axis[0]
    else:
        # need to reorder dims
        ordered_joint_eulers = []
        for axis in rot_order:
            i = rot_channels.index(axis + "rotation")
            ordered_joint_eulers.append(joint_eulers[..., i])
        joint_eulers = np.stack(ordered_joint_eulers, axis=-1)

    rotations = Rotation.from_euler(rot_order, joint_eulers.reshape(-1, 3))
    if return_quat:
        joint_rots = rotations.as_quat(scalar_first=True).reshape(joint_eulers.shape[:-1] + (4,))
    else:
        joint_rots = rotations.as_matrix().reshape(joint_eulers.shape[:-1] + (3, 3))

    return root_trans, joint_rots


def load_csv_g1_animation(
    fname: str,
    skeleton: Skeleton,
    csv_format: str = "retargeted_from_gmr",
    xml_path: str = _DEFAULT_G1_XML,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a csv file that's in the format of mujoco qpos (32 joints, dof 29) return root
    translation and joint rotation matrics for the motion sequence.

    Args:
        fname (str): full path of the csv file
        skeleton (Skeleton): Skeleton object initialized by the bvh file, containing the neutral skeleton
        csv_format (str): format of the csv file ["retargeted_from_gmr", "retargeted_from_unitree"]
        xml_path (str): full path of the xml file for the skeleton
    Returns:
        torch.Tensor: [T, 3] root translations
        torch.Tensor: [T, J, 3, 3] joint rotation matrix
    """
    assert csv_format in ["retargeted_from_gmr", "retargeted_from_unitree"]
    import xml.etree.ElementTree as ET

    with open(fname) as f:
        csv_raw_data = f.readlines()
    if csv_format == "retargeted_from_gmr":
        csv_raw_data = np.array(
            [np.array([float(i) for i in row.split(",")[1:]]) for row in csv_raw_data[1:]]
        )  # drop the title row, and the frame column (first column)
        root_trans = csv_raw_data[:, :3]  # cm since it will be converted back to m outside this func
        xyz = csv_raw_data[:, 3:6]
        root_rot = Rotation.from_euler("xyz", xyz, degrees=True)

    elif csv_format == "retargeted_from_unitree":
        csv_raw_data = np.array([np.array([float(i) for i in row.split(",")]) for row in csv_raw_data])
        root_trans = (
            np.array(csv_raw_data[:, :3]) * 100
        )  # m -> cm since it will be converted back to m outside this func
        root_rot = Rotation.from_quat(csv_raw_data[:, 3:7], scalar_first=False)  # [x, y, z, w] is the order in csv file

    # there are two transformations from the mujoco coordinate to the ardy coordinate (both are right hand system):
    # 1) the mujoco is x facing, and ardy is y facing in the mujoco coordinate (or z facing in the ardy system).
    # 2) the source csv uses z up, and the ardy uses y up;
    R_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
    x_forward_to_y_forward = Rotation.from_euler("z", -90, degrees=True)
    mujoco_to_ardy = R_zup_to_yup * x_forward_to_y_forward

    # for root transform, we do the similar transformation for the root rotations.
    root_trans = mujoco_to_ardy.apply(root_trans)
    root_rot = mujoco_to_ardy * root_rot * mujoco_to_ardy.inv()

    # get the joint rotations from the csv file
    rotations = Rotation.identity(csv_raw_data.shape[0] * len(skeleton.bones))
    joint_rots = rotations.as_matrix().reshape(csv_raw_data.shape[0], len(skeleton.bones), 3, 3)
    joint_rots[:, 0] = root_rot.as_matrix()  # the rotation of pelvis

    # load the mujoco xml file for joint axis; find all joint, get name and axis
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_classes = [x for x in tree.findall(".//default") if "class" in x.attrib]
    joint_axes = dict()
    for xml_class in xml_classes:
        j = xml_class.findall("joint")
        if j:
            joint_axes[xml_class.get("class")] = j[0].get("axis")

    parent_map = {child: parent for parent in root.iter() for child in parent}

    for joint_id_in_csv, joint in enumerate(root.find("worldbody").findall(".//joint")):  # skip the base joint
        # the order in the xml file is the order of the data shows up; note that in the xml file, the joint ends with
        # "_joint", but the skeleton.get_bones_names() joints end with "_skel"
        assert joint.get("name").endswith("_joint")
        joint_name_in_skeleton = joint.get("name").replace("_joint", "_skel")
        assert joint_name_in_skeleton in skeleton.get_bones_names()
        axis_values = [float(x) for x in (joint.get("axis") or joint_axes[joint.get("class")]).split(" ")]
        assert sum(axis_values) == 1 and all(x in [0, 1] for x in axis_values), (
            f"Invalid axis: {axis_values}: one and only one of the axis values should be 1; others should be 0s."
        )

        # the mapped axis in the ardy's g1 skeleton space is calculated as bones_axis = mujoco_to_ardy.apply(axis_values)
        # [1, 0, 0] -> [0, 0, 1]; [0, 1, 0] -> [1, 0, 0]; [0, 0, 1] -> [0, 1, 0]
        axis_in_ardy = ["x", "y", "z"][np.argmax(axis_values)]
        if csv_format == "retargeted_from_gmr":
            joint_dof = csv_raw_data[:, joint_id_in_csv + 6] * np.pi / 180
        elif csv_format == "retargeted_from_unitree":
            joint_dof = csv_raw_data[:, joint_id_in_csv + 7]
        else:
            raise ValueError(f"Unknown csv format: {csv_format}")

        # Single-axis sequence needs angles of shape (N, 1) under SciPy's array-API
        # from_euler, which reads the last axis as the axis count (not a bare (N,)).
        joint_rot = Rotation.from_euler(axis_in_ardy, joint_dof[:, None], degrees=False)

        body = parent_map[joint]
        if "quat" in body.attrib:
            joint_extra_rot = Rotation.from_quat(
                [float(x) for x in body.get("quat").strip().split(" ")],
                scalar_first=True,
            )
            joint_rot = joint_extra_rot * joint_rot

        joint_rots[:, skeleton.get_bones_names().index(joint_name_in_skeleton)] = (
            mujoco_to_ardy * joint_rot * mujoco_to_ardy.inv()
        ).as_matrix()

    return root_trans, joint_rots
