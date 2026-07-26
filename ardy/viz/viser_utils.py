# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple, Union
import time

import numpy as np
import torch
import trimesh
import viser
import viser.transforms as tf

from scipy.interpolate import interp1d

from ardy.assets import skeleton_asset_path
from ardy.skeleton.kinematics import batch_rigid_transform
from ardy.skeleton import (
    CoreSkeleton27,
    G1Skeleton34,
    SOMASkeleton30,
    SOMASkeleton77,
    SkeletonBase,
)
from ardy.motion_rep.smooth_root import get_smooth_root_pos
from ardy.skeleton.transforms import global_rots_to_local_rots
from ardy.tools import to_numpy, to_torch
from ardy.viz.core_skin import CoreSkin
from ardy.viz.g1_rig import G1MeshRig
from ardy.viz.soma_skin import SOMASkin


_G1_JOINT_AXIS_INDEX_CACHE: Optional[dict[str, int]] = None


def _get_g1_joint_axis_indices() -> dict[str, int]:
    """Return a map from G1 joint names to a single rotation axis index."""
    global _G1_JOINT_AXIS_INDEX_CACHE
    if _G1_JOINT_AXIS_INDEX_CACHE is not None:
        return _G1_JOINT_AXIS_INDEX_CACHE

    xml_path = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    if not os.path.exists(xml_path):
        _G1_JOINT_AXIS_INDEX_CACHE = {}
        return _G1_JOINT_AXIS_INDEX_CACHE

    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_axes = {}
    for xml_class in tree.findall(".//default"):
        if "class" not in xml_class.attrib:
            continue
        joint_nodes = xml_class.findall("joint")
        if joint_nodes:
            joint_axes[xml_class.get("class")] = joint_nodes[0].get("axis")

    # mujoco (z-up, x-forward) -> ardy (y-up, z-forward)
    mujoco_to_ardy = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    axis_indices_by_name: dict[str, int] = {}
    for joint in root.find("worldbody").findall(".//joint"):
        axis_str = joint.get("axis") or joint_axes.get(joint.get("class"))
        if axis_str is None:
            continue
        axis_vals = np.array([float(x) for x in axis_str.split()], dtype=np.float64)
        if not np.any(axis_vals):
            continue
        axis_ardy = mujoco_to_ardy @ axis_vals
        axis_idx = int(np.argmax(np.abs(axis_ardy)))
        axis_indices_by_name[joint.get("name").replace("_joint", "_skel")] = axis_idx

    _G1_JOINT_AXIS_INDEX_CACHE = axis_indices_by_name
    return _G1_JOINT_AXIS_INDEX_CACHE


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross products: skew(v) @ x == np.cross(v, x)."""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64)


def _rotation_matrix_from_two_vec(v_from: np.ndarray, v_to: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return R such that R @ v_from ~= v_to (both treated as 3D vectors).

    Uses a Rodrigues-style construction, with special handling for near-parallel and near-opposite
    vectors for numerical stability.
    """
    a = np.asarray(v_from, dtype=np.float64).reshape(3)
    b = np.asarray(v_to, dtype=np.float64).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = b / nb

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))  # cos(theta)
    if c > 1.0 - eps:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + eps:
        # 180 deg rotation about any axis orthogonal to a:
        # R = -I + 2 * uu^T, where u is a unit axis orthogonal to a.
        axis_seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a, axis_seed))) > 0.9:
            axis_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(a, axis_seed)
        u = u / np.linalg.norm(u).clip(min=eps)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(u, u)

    v = np.cross(a, b)
    s2 = float(np.dot(v, v))  # ||v||^2 == sin^2(theta)
    K = _skew(v)
    # R = I + K + K^2 * ((1 - c) / s^2)
    return np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / s2)


# TODO: should things in here by kept on cpu with numpy to avoid latency when interacting with UI?
#       the downside is we need torch/GPU for FK

# Cache arrow meshes to avoid recreating them repeatedly
_CACHED_ARROW_BASE = None
_CACHED_ARROW_HEAD = None


def _get_cached_arrow_meshes():
    """Get cached arrow base and head meshes (vertices and faces only)."""
    global _CACHED_ARROW_BASE, _CACHED_ARROW_HEAD
    if _CACHED_ARROW_BASE is None:
        arrow_base_mesh = trimesh.creation.cylinder(radius=0.01, height=0.2)
        arrow_head_mesh = trimesh.creation.cone(radius=0.03, height=0.05)
        _CACHED_ARROW_BASE = {
            "vertices": arrow_base_mesh.vertices.copy(),
            "faces": arrow_base_mesh.faces.copy(),
        }
        _CACHED_ARROW_HEAD = {
            "vertices": arrow_head_mesh.vertices.copy(),
            "faces": arrow_head_mesh.faces.copy(),
        }
    return _CACHED_ARROW_BASE, _CACHED_ARROW_HEAD


class WaypointMesh:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        position: np.ndarray,
        heading: Optional[np.ndarray] = None,
        color: Optional[Tuple[int, int, int]] = (255, 0, 0),
        add_annulus: bool = True,
    ):
        self.server = server
        self.color = color
        self.base_name = name  # Store base name to remove parent folder later

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.025)

        z_to_y_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        self.sphere = self.server.scene.add_mesh_simple(
            name=f"{name}/sphere",
            vertices=sphere.vertices,
            faces=sphere.faces,
            position=position,
            color=color,
        )

        if add_annulus:
            annulus = trimesh.creation.annulus(r_min=0.1, r_max=0.2, height=0.005)
            annulus_vertices = annulus.vertices @ z_to_y_up
            self.annulus = self.server.scene.add_mesh_simple(
                name=f"{name}/annulus",
                vertices=annulus_vertices,
                faces=annulus.faces,
                position=position,
                color=color,
            )
        else:
            self.annulus = None

        self.arrow_base = None
        self.arrow_head = None
        if heading is not None:
            assert heading.shape == (2,), "Heading must be a 2D vector"
            heading_norm = heading / np.linalg.norm(heading)
            heading_scaled = 0.2 * heading_norm
            heading_3d = np.array([heading_scaled[0], 0, heading_scaled[1]])

            # Calculate rotation to align Y-axis (default cylinder/cone orientation) with heading
            # Rotation angle around Y-axis
            angle = np.arctan2(heading_norm[0], heading_norm[1])  # heading = [cos, sin] -> angle
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Rotation matrix around Y-axis
            rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

            # Use cached arrow meshes
            arrow_base_cache, arrow_head_cache = _get_cached_arrow_meshes()

            # Rotate vertices to align with heading direction
            arrow_base_vertices = arrow_base_cache["vertices"] @ rot_y.T
            arrow_head_vertices = arrow_head_cache["vertices"] @ rot_y.T

            self.arrow_base = self.server.scene.add_mesh_simple(
                name=f"{name}/arrow_base",
                vertices=arrow_base_vertices,
                faces=arrow_base_cache["faces"],
                position=position + (heading_3d / 2),
                color=color,
            )
            self.arrow_head = self.server.scene.add_mesh_simple(
                name=f"{name}/arrow_head",
                vertices=arrow_head_vertices,
                faces=arrow_head_cache["faces"],
                position=position + heading_3d,
                color=color,
            )

    def update_position(self, position: np.ndarray, heading: Optional[np.ndarray] = None):
        self.sphere.position = position
        if self.annulus is not None:
            self.annulus.position = position
        if heading is not None:
            assert heading.shape == (2,), "Heading must be a 2D vector"
            heading_norm = heading / np.linalg.norm(heading)
            heading_scaled = 0.2 * heading_norm
            heading_3d = np.array([heading_scaled[0], 0, heading_scaled[1]])

            # Calculate rotation to align Y-axis with heading
            angle = np.arctan2(heading_norm[0], heading_norm[1])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

            # Remove old arrow meshes if they exist
            if self.arrow_base is not None:
                self.server.scene.remove_by_name(self.arrow_base.name)
            if self.arrow_head is not None:
                self.server.scene.remove_by_name(self.arrow_head.name)

            # Use cached arrow meshes
            arrow_base_cache, arrow_head_cache = _get_cached_arrow_meshes()

            # Rotate vertices to align with heading direction
            arrow_base_vertices = arrow_base_cache["vertices"] @ rot_y.T
            arrow_head_vertices = arrow_head_cache["vertices"] @ rot_y.T

            # Get base name from sphere (e.g., "/waypoint_0/sphere" -> "/waypoint_0")
            base_name = self.sphere.name.rsplit("/", 1)[0]

            self.arrow_base = self.server.scene.add_mesh_simple(
                name=f"{base_name}/arrow_base",
                vertices=arrow_base_vertices,
                faces=arrow_base_cache["faces"],
                position=position + (heading_3d / 2),
                color=self.color,
            )
            self.arrow_head = self.server.scene.add_mesh_simple(
                name=f"{base_name}/arrow_head",
                vertices=arrow_head_vertices,
                faces=arrow_head_cache["faces"],
                position=position + heading_3d,
                color=self.color,
            )

    def clear(self):
        """Remove all waypoint meshes and the parent folder from the scene."""
        # self.server.scene.remove_by_name(self.sphere.name)
        # if self.annulus is not None:
        #     self.server.scene.remove_by_name(self.annulus.name)
        # if self.arrow_base is not None:
        #     self.server.scene.remove_by_name(self.arrow_base.name)
        # if self.arrow_head is not None:
        #     self.server.scene.remove_by_name(self.arrow_head.name)
        # Remove the parent folder to prevent empty folders in scene tree
        try:
            # print(f"Removing parent folder {self.base_name}")
            self.server.scene.remove_by_name(self.base_name)
        except Exception as e:
            print(f"Error removing parent folder {self.base_name}: {e}")
            pass  # Parent folder might have already been removed


class VelocityArrowMesh:
    """Visualizes root velocity as an arrow (line segment + cone head)."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        color: tuple = (50, 150, 255),  # Default blue color (RGB)
    ):
        """Initialize velocity arrow visualization.

        Args:
            name: str, base name for the arrow components
            server: viser.ViserServer, server to add the arrow to
            skeleton: SkeletonBase, skeleton to get root index from
            color: tuple, RGB color tuple (0-255), default is blue
        """
        self.name = name
        self.server = server
        self.skeleton = skeleton
        self.color = color

        # Arrow components
        self.arrow_line = None  # Line segment
        self.arrow_cone = None  # Cone head
        self.should_show = False  # Track if arrow should be visible based on velocity magnitude

    def update(
        self,
        root_velocity: Optional[Union[np.ndarray, torch.Tensor]],
        root_pos: Union[np.ndarray, torch.Tensor],
        visible: bool = True,
    ):
        """Update the velocity arrow visualization.

        Args:
            root_velocity: Optional[Union[np.ndarray, torch.Tensor]], [3] root joint velocity (x, y, z) in m/s
            root_pos: Union[np.ndarray, torch.Tensor], [3] root position
            visible: bool, whether the arrow should be visible (controlled by skeleton visibility)
        """
        if root_velocity is None:
            # Hide arrow if no velocity provided
            self.should_show = False
            if self.arrow_line is not None:
                self.arrow_line.visible = False
                self.arrow_cone.visible = False
            return

        # Convert to numpy if tensor
        if isinstance(root_velocity, torch.Tensor):
            root_velocity = root_velocity.detach().cpu().numpy()
        if isinstance(root_pos, torch.Tensor):
            root_pos = root_pos.detach().cpu().numpy()

        # Project velocity to XZ plane
        velocity_xz = np.array([root_velocity[0], 0.0, root_velocity[2]])
        velocity_magnitude = np.linalg.norm(velocity_xz)

        # Only show arrow if velocity is significant (> 0.1 m/s)
        if velocity_magnitude <= 0.1:
            self.should_show = False
            if self.arrow_line is not None:
                self.arrow_line.visible = False
                self.arrow_cone.visible = False
            return

        # Arrow should be shown
        self.should_show = True

        # Calculate arrow geometry
        root_2d_pos = np.array([root_pos[0], 0.0, root_pos[2]])  # Project to ground
        velocity_dir = velocity_xz / velocity_magnitude
        arrow_length = velocity_magnitude / 4.0
        arrow_end = root_2d_pos + velocity_dir * arrow_length

        # Calculate rotation quaternion for arrow head
        from_vec = np.array([0.0, 0.0, 1.0])
        to_vec = velocity_dir
        rot_mat = _rotation_matrix_from_two_vec(from_vec, to_vec)
        quat = tf.SO3.from_matrix(rot_mat).wxyz

        # Create or update arrow components
        if self.arrow_line is None:
            # Create line segment
            self.arrow_line = self.server.scene.add_line_segments(
                name=f"{self.name}/velocity_line",
                points=np.array([[root_2d_pos, arrow_end]]),
                colors=self.color,
                line_width=3.0,
            )

            # Create cone head
            arrow_head = trimesh.creation.cone(radius=0.04, height=0.1)
            self.arrow_cone = self.server.scene.add_mesh_simple(
                name=f"{self.name}/velocity_cone",
                vertices=arrow_head.vertices,
                faces=arrow_head.faces,
                color=self.color,
                position=arrow_end,
                wxyz=quat,
            )
        else:
            # Update existing components atomically
            # Hide both during update
            old_line_visible = self.arrow_line.visible
            old_cone_visible = self.arrow_cone.visible
            self.arrow_line.visible = False
            self.arrow_cone.visible = False

            # Update geometry
            self.arrow_line.points = np.array([[root_2d_pos, arrow_end]])
            self.arrow_cone.position = arrow_end - 0.05 * velocity_dir
            self.arrow_cone.wxyz = quat

            # Restore visibility
            self.arrow_line.visible = old_line_visible
            self.arrow_cone.visible = old_cone_visible

        # Set visibility based on skeleton visibility and should_show
        self.arrow_line.visible = visible and self.should_show
        self.arrow_cone.visible = visible and self.should_show

    def set_visibility(self, visible: bool):
        """Set visibility of the velocity arrow."""
        if self.arrow_line is not None:
            self.arrow_line.visible = visible and self.should_show
        if self.arrow_cone is not None:
            self.arrow_cone.visible = visible and self.should_show

    def clear(self):
        """Remove the velocity arrow from the scene."""
        if self.arrow_line is not None:
            self.server.scene.remove_by_name(self.arrow_line.name)
            self.arrow_line = None
        if self.arrow_cone is not None:
            self.server.scene.remove_by_name(self.arrow_cone.name)
            self.arrow_cone = None


class SkeletonMesh:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        joint_color: Optional[Tuple[float, float, float] | np.ndarray] = (
            255,
            235,
            0,
        ),
        bone_color: Optional[Tuple[float, float, float] | np.ndarray] = (
            27,
            106,
            0,
        ),
        starting_joints_pos: Optional[torch.Tensor] = None,
        show_root_2d_projection: bool = False,
    ):
        """
        name: str, name of the skeleton mesh
        server: viser.ViserServer, server to add the skeleton mesh to
        skeleton: SkeletonBase, skeleton to visualize (must be a CoreSkeleton27)
        joint_color: Optional[Tuple[float, float, float] | np.ndarray], color of the joints, either (3,) or (J, 3)
        bone_color: Optional[Tuple[float, float, float] | np.ndarray], color of the bones, either (3,) or (J-1, 3)
        starting_joints_pos: Optional[torch.Tensor], starting joint positions (if None, will use neutral pose)
        show_root_2d_projection: bool, whether to show the 2D root projection as a blue sphere
        """
        self.server = server
        self.skeleton = skeleton
        self.show_root_2d_projection = show_root_2d_projection
        joint_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.02)
        bone_mesh = trimesh.creation.cylinder(radius=0.01, height=1.0)

        init_joints_pos = skeleton.neutral_joints.clone()
        self.num_joints = init_joints_pos.shape[0]
        num_bones = self.num_joints - 1
        non_root_bones = [
            joint_name
            for joint_name, parent_name in self.skeleton.bone_order_names_with_parents
            if parent_name is not None
        ]
        self.bone_to_idx = {bone_name: idx for idx, bone_name in enumerate(non_root_bones)}

        # initialize meshes
        init_joints_wxyzs = np.concatenate([np.ones((self.num_joints, 1)), np.zeros((self.num_joints, 3))], axis=1)
        if isinstance(joint_color, tuple):
            self.joint_colors = np.full((self.num_joints, 3), joint_color)
        elif isinstance(joint_color, np.ndarray):
            assert joint_color.shape == (
                self.num_joints,
                3,
            ), "Joint colors must be (J, 3)"
            self.joint_colors = joint_color
        self.joints_batched_mesh = server.scene.add_batched_meshes_simple(
            f"{name}/joints",
            vertices=joint_mesh.vertices,
            faces=joint_mesh.faces,
            batched_wxyzs=init_joints_wxyzs,
            batched_positions=np.zeros((self.num_joints, 3)),
            batched_scales=np.ones((self.num_joints, 3)),
            batched_colors=self.joint_colors,
        )
        init_bones_wxyzs = np.concatenate([np.ones((num_bones, 1)), np.zeros((num_bones, 3))], axis=1)
        if isinstance(bone_color, tuple):
            bone_color = np.full((num_bones, 3), bone_color)
        elif isinstance(bone_color, np.ndarray):
            assert bone_color.shape == (num_bones, 3), "Bone colors must be (J-1, 3)"
            bone_color = bone_color
        self.bones_batched_mesh = server.scene.add_batched_meshes_simple(
            f"{name}/bones",
            vertices=bone_mesh.vertices,
            faces=bone_mesh.faces,
            batched_wxyzs=init_bones_wxyzs,
            batched_positions=np.zeros((num_bones, 3)),
            batched_scales=np.ones((num_bones, 3)),
            batched_colors=bone_color,
        )

        # Initialize 2D root projection sphere (blue)
        if self.show_root_2d_projection:
            root_2d_mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
            self.root_2d_sphere = server.scene.add_mesh_simple(
                f"{name}/root_2d_projection",
                vertices=root_2d_mesh.vertices,
                faces=root_2d_mesh.faces,
                color=(50, 150, 255),  # Blue color
                position=(0, 0, 0),
            )
        else:
            self.root_2d_sphere = None

        # Initialize root velocity arrow (visualize XZ projection)
        self.velocity_arrow_mesh = VelocityArrowMesh(
            name=name,
            server=server,
            skeleton=skeleton,
        )

        # used if precomputed meshes are used
        self.mesh_info_cache = None

        if starting_joints_pos is not None:
            self.set_pose(starting_joints_pos)
        else:
            # set them to neutral pose
            min_height = init_joints_pos[:, 1].min().item()
            init_joints_pos[:, 1] -= min_height  # move to be on ground
            self.set_pose(init_joints_pos)

    def compute_single_pose(self, joints_pos: np.ndarray):
        """Compute the mesh for a single frame.

        joints_pos: [J, 3] global joint positions
        """
        # compute bone transforms
        new_batched_positions = np.zeros((self.skeleton.nbjoints - 1, 3))
        new_batched_wxyzs = np.zeros((self.skeleton.nbjoints - 1, 4))
        new_batched_scales = np.ones((self.skeleton.nbjoints - 1, 3))
        for joint_name, parent_name in self.skeleton.bone_order_names_with_parents:
            if parent_name is None:
                continue
            joint_idx = self.skeleton.bone_index[joint_name]
            parent_idx = self.skeleton.bone_index[parent_name]
            joint_pos = joints_pos[joint_idx]
            parent_pos = joints_pos[parent_idx]

            bone_pos = (joint_pos + parent_pos) / 2.0
            bone_scale = np.linalg.norm(joint_pos - parent_pos)
            if bone_scale < 1e-8:
                bone_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                bone_dir = (joint_pos - parent_pos) / bone_scale
                R = _rotation_matrix_from_two_vec(np.array([0.0, 0.0, 1.0], dtype=np.float64), bone_dir)
                bone_wxyz = tf.SO3.from_matrix(R).wxyz

            bone_idx = self.bone_to_idx[joint_name]
            new_batched_positions[bone_idx] = bone_pos
            new_batched_wxyzs[bone_idx] = bone_wxyz
            new_batched_scales[bone_idx] = np.array([1.0, 1.0, bone_scale], dtype=float)

        return new_batched_positions, new_batched_wxyzs, new_batched_scales

    def precompute_mesh_info(self, joints_pos: torch.Tensor):
        """Precompute the meshes for all frames at once.

        joints_pos: [T, J, 3] global joint positions
        """
        joints_pos = joints_pos.cpu().numpy()
        # compute bone transforms
        num_frames = joints_pos.shape[0]
        self.mesh_info_cache = {
            "positions": np.zeros((num_frames, self.skeleton.nbjoints - 1, 3)),
            "wxyzs": np.zeros((num_frames, self.skeleton.nbjoints - 1, 4)),
            "scales": np.ones((num_frames, self.skeleton.nbjoints - 1, 3)),
        }
        for i in range(num_frames):
            new_batched_positions, new_batched_wxyzs, new_batched_scales = self.compute_single_pose(joints_pos[i])
            self.mesh_info_cache["positions"][i] = new_batched_positions
            self.mesh_info_cache["wxyzs"][i] = new_batched_wxyzs
            self.mesh_info_cache["scales"][i] = new_batched_scales

    def update_mesh_info_cache(self, joints_pos: torch.Tensor, frame_idx: int):
        """Update the mesh info cache for the given frame.

        joints_pos: [J, 3] global joint positions
        """
        assert self.mesh_info_cache is not None
        new_batched_positions, new_batched_wxyzs, new_batched_scales = self.compute_single_pose(
            joints_pos.cpu().numpy()
        )
        self.mesh_info_cache["positions"][frame_idx] = new_batched_positions
        self.mesh_info_cache["wxyzs"][frame_idx] = new_batched_wxyzs
        self.mesh_info_cache["scales"][frame_idx] = new_batched_scales

    def set_pose(
        self,
        joints_pos: torch.Tensor,
        foot_contacts: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
        root_velocity: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        joints_pos: [J, 3] global joint positions
        foot_contacts: [4] contact labels for left heel/toe and right heel/toe, 1 for in contact
        frame_idx: int, index of the frame to set the pose for (only needed if precomputed mesh info is used)
        root_velocity: Optional[Union[np.ndarray, torch.Tensor]], [3] root joint velocity (x, y, z) in m/s
        """
        self.cur_joints_pos = joints_pos
        joints_pos = joints_pos.cpu().numpy()

        if self.mesh_info_cache is not None:
            assert frame_idx is not None
            new_batched_positions = self.mesh_info_cache["positions"][frame_idx]
            new_batched_wxyzs = self.mesh_info_cache["wxyzs"][frame_idx]
            new_batched_scales = self.mesh_info_cache["scales"][frame_idx]
        else:
            new_batched_positions, new_batched_wxyzs, new_batched_scales = self.compute_single_pose(joints_pos)

        # update meshes
        self.bones_batched_mesh.batched_positions = new_batched_positions
        self.bones_batched_mesh.batched_wxyzs = new_batched_wxyzs
        self.bones_batched_mesh.batched_scales = new_batched_scales
        # directly set joint positions
        self.joints_batched_mesh.batched_positions = joints_pos

        # update 2D root projection sphere
        if self.root_2d_sphere is not None:
            root_pos = joints_pos[self.skeleton.root_idx]
            root_2d_pos = np.array([root_pos[0], 0.0, root_pos[2]])
            self.root_2d_sphere.position = root_2d_pos

        # Update root velocity arrow visualization (XZ projection)
        root_pos = joints_pos[self.skeleton.root_idx]
        skeleton_visible = self.joints_batched_mesh.visible
        self.velocity_arrow_mesh.update(
            root_velocity=root_velocity,
            root_pos=root_pos,
            visible=skeleton_visible,
        )

        # update colors for foot contacts
        if foot_contacts is not None:
            cur_joint_colors = self.joint_colors.copy()
            foot_contacts = foot_contacts.bool().cpu().numpy().astype(bool)
            foot_joints = np.array(self.skeleton.foot_joint_idx, dtype=int)
            contact_idx = foot_joints[foot_contacts]
            cur_joint_colors[contact_idx] = (160, 32, 240)
            self.joints_batched_mesh.batched_colors = cur_joint_colors
        else:
            self.joints_batched_mesh.batched_colors = self.joint_colors

    def set_visibility(self, visible: bool):
        self.joints_batched_mesh.visible = visible
        self.bones_batched_mesh.visible = visible
        if self.root_2d_sphere is not None:
            self.root_2d_sphere.visible = visible
        # Update velocity arrow visibility
        self.velocity_arrow_mesh.set_visibility(visible)

    def get_pose(self) -> np.ndarray:
        return self.cur_joints_pos

    def clear(self):
        names = [mesh.name for mesh in [self.joints_batched_mesh, self.bones_batched_mesh]]
        for name in names:
            self.server.scene.remove_by_name(name)
        if self.root_2d_sphere is not None:
            self.server.scene.remove_by_name(self.root_2d_sphere.name)
        # Clear velocity arrow
        self.velocity_arrow_mesh.clear()


LIGHT_THEME = dict(
    mesh=(152, 189, 255),  # (90, 200, 255) - original viser blue
)

DARK_THEME = dict(
    mesh=(60, 85, 130),
)


class Character:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        skeleton: SkeletonBase,
        create_skeleton_mesh: bool = True,
        create_skinned_mesh: bool = True,
        visible_skeleton: bool = False,
        visible_skinned_mesh: bool = True,
        skinned_mesh_opacity: float = 1.0,
        show_foot_contacts: bool = True,
        dark_mode: bool = False,
        mesh_mode: str = "core_skin",
        g1_mesh_dir: Optional[str] = None,
        show_root_2d_projection: bool = False,
    ):
        self.server = server
        self.name = name
        self.skeleton = skeleton
        self.cur_joints_pos = None
        self.cur_joints_rot = None
        self.cur_foot_contacts = None

        self.skeleton_mesh = None
        self.show_foot_contacts = show_foot_contacts
        if create_skeleton_mesh:
            self.skeleton_mesh = SkeletonMesh(
                f"/{name}/skeleton",
                server,
                skeleton,
                show_root_2d_projection=show_root_2d_projection,
            )
            # init with default rest pose
            self.cur_joints_pos = self.skeleton_mesh.get_pose()
            self.skeleton_mesh.set_visibility(visible_skeleton)

        self.skinned_mesh = None
        self.g1_mesh_rig = None
        self.skin = None
        self.mesh_mode = mesh_mode
        if create_skinned_mesh:
            if isinstance(self.skeleton, CoreSkeleton27) and mesh_mode == "core_skin":
                self.skin = CoreSkin(self.skeleton)
                self.skinned_mesh = server.scene.add_mesh_simple(
                    f"/{name}/simple_skinned",
                    vertices=self.skin.bind_vertices.cpu().numpy(),
                    faces=self.skin.faces.cpu().numpy(),
                    opacity=None,
                    color=LIGHT_THEME["mesh"] if not dark_mode else DARK_THEME["mesh"],
                    wireframe=False,
                    visible=False,
                )
                self.skinned_verts_cache = None

                bind_pos = self.skeleton.neutral_joints.clone()
                min_height = bind_pos[:, 1].min().item()
                bind_pos[:, 1] -= min_height  # move to be on ground
                bind_rotmat = torch.eye(3, device=bind_pos.device).repeat(bind_pos.shape[0], 1, 1)
                self.set_pose(bind_pos, bind_rotmat)
                self.skinned_mesh.visible = True  # avoid blinking
                self.set_skinned_mesh_visibility(visible_skinned_mesh)
                self.set_skinned_mesh_opacity(skinned_mesh_opacity)
            elif isinstance(self.skeleton, (SOMASkeleton30, SOMASkeleton77)) and mesh_mode == "soma_skin":
                self.skin = SOMASkin(self.skeleton)
                self.skinned_mesh = server.scene.add_mesh_simple(
                    f"/{name}/simple_skinned",
                    vertices=self.skin.bind_vertices.cpu().numpy(),
                    faces=self.skin.faces.cpu().numpy(),
                    opacity=None,
                    color=LIGHT_THEME["mesh"] if not dark_mode else DARK_THEME["mesh"],
                    wireframe=False,
                    visible=False,
                )
                self.skinned_verts_cache = None

                bind_pos = self.skeleton.neutral_joints.clone()
                min_height = bind_pos[:, 1].min().item()
                bind_pos[:, 1] -= min_height  # move to be on ground
                bind_rotmat = torch.eye(3, device=bind_pos.device).repeat(bind_pos.shape[0], 1, 1)
                self.set_pose(bind_pos, bind_rotmat)
                self.skinned_mesh.visible = True  # avoid blinking
                self.set_skinned_mesh_visibility(visible_skinned_mesh)
                self.set_skinned_mesh_opacity(skinned_mesh_opacity)
            elif isinstance(self.skeleton, G1Skeleton34) and mesh_mode == "g1_stl":
                if g1_mesh_dir is None:
                    g1_mesh_dir = os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "assets",
                        "skeletons",
                        "g1skel34",
                        "meshes",
                        "g1",
                    )
                    g1_mesh_dir = os.path.abspath(g1_mesh_dir)
                if not os.path.exists(g1_mesh_dir):
                    print(f"G1 mesh directory not found: {g1_mesh_dir}")
                self.g1_mesh_rig = G1MeshRig(
                    name,
                    server,
                    self.skeleton,
                    g1_mesh_dir,
                    DARK_THEME["mesh"] if dark_mode else LIGHT_THEME["mesh"],
                )
                init_joints_pos = self.skeleton.neutral_joints.clone()
                min_height = init_joints_pos[:, 1].min().item()
                init_joints_pos[:, 1] -= min_height  # move to be on ground
                init_joints_rot = torch.eye(3, device=init_joints_pos.device).repeat(init_joints_pos.shape[0], 1, 1)
                self.set_pose(init_joints_pos, init_joints_rot)
                self.set_skinned_mesh_visibility(visible_skinned_mesh)
                self.set_skinned_mesh_opacity(skinned_mesh_opacity)
            else:
                raise ValueError(
                    "Unsupported mesh mode for skeleton type: "
                    f"{type(self.skeleton).__name__} with mesh_mode={mesh_mode}"
                )

    def change_theme(self, is_dark_mode):
        color = DARK_THEME["mesh"] if is_dark_mode else LIGHT_THEME["mesh"]
        if self.skinned_mesh is not None:
            self.skinned_mesh.color = color
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_color(color)

    def set_skeleton_visibility(self, visible: bool):
        if self.skeleton_mesh is not None:
            self.skeleton_mesh.set_visibility(visible)

    def set_show_foot_contacts(self, show: bool):
        self.show_foot_contacts = show

    def set_skinned_mesh_visibility(self, visible: bool):
        if self.skinned_mesh is not None:
            self.skinned_mesh.visible = visible
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_visibility(visible)

    def set_skinned_mesh_opacity(self, opacity: float):
        if self.skinned_mesh is not None:
            self.skinned_mesh.opacity = opacity
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_opacity(opacity)

    def set_skinned_mesh_wireframe(self, wireframe: bool):
        if self.skinned_mesh is not None:
            self.skinned_mesh.wireframe = wireframe
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_wireframe(wireframe)

    def precompute_skinning(self, joints_pos: torch.Tensor, joints_rot: torch.Tensor):
        """If using simple skinning, we can precompute the skinning for all frames at once.

        joints_pos: [T, J, 3] global joint positions
        joints_rot: [T, J, 3, 3] global joint rotation matrices
        """
        assert self.skin is not None
        self.skinned_verts_cache = self.skin.skin(joints_rot, joints_pos, rot_is_global=True).cpu().numpy()

    def update_skinning_cache(self, joints_pos: torch.Tensor, joints_rot: torch.Tensor, frame_idx: int):
        """Update the skinning cache for the given frame.

        joints_pos: [J, 3] global joint positions
        joints_rot: [J, 3, 3] global joint rotation matrices
        frame_idx: int, index of the frame to update the cache for
        """
        if self.skinned_verts_cache is None:
            return

        new_skinned_verts = self.skin.skin(joints_rot[None], joints_pos[None], rot_is_global=True)[0].cpu().numpy()
        self.skinned_verts_cache[frame_idx] = new_skinned_verts

    def set_pose(
        self,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
        foot_contacts: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
        root_velocity: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        joints_pos: [J, 3] global joint positions
        joints_rot: [J, 3, 3] global joint rotation matrices
        frame_idx: int, index of the frame to set the pose for (only needed if precomputed skinning is used)
        root_velocity: Optional[Union[np.ndarray, torch.Tensor]], [3] root joint velocity (x, y, z) in m/s
        """
        if self.skeleton_mesh is not None:
            cur_foot_contacts = foot_contacts if self.show_foot_contacts else None
            self.skeleton_mesh.set_pose(
                joints_pos,
                foot_contacts=cur_foot_contacts,
                frame_idx=frame_idx,
                root_velocity=root_velocity,
            )
            self.cur_foot_contacts = cur_foot_contacts

        if self.skinned_mesh is not None:
            if self.skinned_verts_cache is not None:
                assert frame_idx is not None
                skinned_verts = self.skinned_verts_cache[frame_idx]
            else:
                skinned_verts = self.skin.skin(joints_rot[None], joints_pos[None], rot_is_global=True)[0].cpu().numpy()

            # update the vertices
            self.skinned_mesh.vertices = skinned_verts
        if self.g1_mesh_rig is not None:
            joints_pos_np = joints_pos.detach().cpu().numpy()
            joints_rot_np = joints_rot.detach().cpu().numpy()
            self.g1_mesh_rig.set_pose(joints_pos_np, joints_rot_np)

        self.cur_joints_pos = joints_pos
        self.cur_joints_rot = joints_rot

    def get_pose(self) -> torch.Tensor:
        return self.cur_joints_pos, self.cur_joints_rot

    def clear(self):
        if self.skeleton_mesh is not None:
            self.skeleton_mesh.clear()
        if self.skinned_mesh is not None:
            self.server.scene.remove_by_name(self.skinned_mesh.name)
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.clear()


class CharacterMotion:
    def __init__(
        self,
        character: Character,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
        foot_contacts: Optional[torch.Tensor] = None,
    ):
        self.character = character
        self.server = character.server
        self.skeleton = character.skeleton
        self.name = character.name

        # [T, J, 3] global joint positions
        self.joints_pos = joints_pos
        # [T, J, 3, 3] global joint rotation matrices
        self.joints_rot = joints_rot
        assert joints_pos.shape[0] == joints_rot.shape[0]
        # keep track of local rots as well for convenience during pose editing
        self.joints_local_rot = global_rots_to_local_rots(joints_rot, self.skeleton)

        self.length = joints_pos.shape[0]
        self.cur_frame_idx = None

        self.foot_contacts = foot_contacts
        if foot_contacts is not None:
            assert foot_contacts.shape[0] == self.length

        self.precompute_mesh_info()

        # gizmos for pose editing
        self.root_translation_gizmo = None
        self.updating_root_translation_gizmo = False
        self.joint_gizmos = None
        self.updating_joint_gizmos = False

    def precompute_mesh_info(self):
        if self.character.skeleton_mesh is not None:
            print("Caching skeleton mesh info...")
            self.character.skeleton_mesh.precompute_mesh_info(self.joints_pos)
        if self.character.skinned_mesh is not None:
            print("Caching skinning info...")
            self.character.precompute_skinning(self.joints_pos, self.joints_rot)

    def set_frame(self, idx: int):
        """Sets the pose of the character to the given frame index."""
        idx = min(idx, self.length - 1)  # clamp to last frame
        cur_foot_contacts = self.foot_contacts[idx] if self.foot_contacts is not None else None
        self.character.set_pose(
            self.joints_pos[idx],
            self.joints_rot[idx],
            frame_idx=idx,
            foot_contacts=cur_foot_contacts,
        )
        self.cur_frame_idx = idx

        # update gizmos if frame has changed due to playback
        cur_root_pos = self.joints_pos[self.cur_frame_idx, self.skeleton.root_idx].clone()
        cur_root_pos[1] = 0.0
        if self.root_translation_gizmo is not None and not self.updating_root_translation_gizmo:
            self.root_translation_gizmo.position = cur_root_pos.cpu().numpy()
        if self.joint_gizmos is not None:
            for i, joint_gizmo in enumerate(self.joint_gizmos):
                if not self.updating_joint_gizmos:
                    joint_gizmo.position = self.joints_pos[self.cur_frame_idx, i].cpu().numpy()
                    joint_gizmo.wxyz = tf.SO3.from_matrix(
                        self.joints_local_rot[self.cur_frame_idx, i].cpu().numpy()
                    ).wxyz

    def update_pose_at_frame(
        self,
        frame_idx: int,
        joints_pos: Optional[torch.Tensor] = None,
        joints_rot: Optional[torch.Tensor] = None,
        joints_local_rot: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ):
        """Overwrites one or more of the pose components at the given frame.

        If only a subset of joints_pos, joints_rot, or joints_local_rot are provided, the other
        components will be updated with FK.
        """
        if joints_pos is not None:
            joints_pos = to_torch(joints_pos, device=self.joints_pos.device, dtype=self.joints_pos.dtype)
            self.joints_pos[frame_idx] = joints_pos
            if joints_local_rot is None and joints_rot is None:
                raise NotImplementedError("No IK to update joint rotations accordingly.")
        if joints_rot is not None:
            joints_rot = to_torch(joints_rot, device=self.joints_rot.device, dtype=self.joints_rot.dtype)
            self.joints_rot[frame_idx] = joints_rot
            if joints_local_rot is None:
                # update local rots from global rots
                self.joints_local_rot[frame_idx] = global_rots_to_local_rots(joints_rot, self.skeleton)
            if joints_pos is None:
                # need to update with FK
                new_posed_joints, _ = batch_rigid_transform(
                    self.joints_local_rot[frame_idx : frame_idx + 1],
                    self.skeleton.neutral_joints[None].to(self.joints_local_rot.device),
                    self.skeleton.joint_parents.to(self.joints_local_rot.device),
                    self.skeleton.root_idx,
                )
                new_posed_joints = (
                    new_posed_joints[0]
                    + self.joints_pos[frame_idx, self.skeleton.root_idx : self.skeleton.root_idx + 1]
                    - self.skeleton.neutral_joints[[self.skeleton.root_idx]]
                )
                self.joints_pos[frame_idx] = new_posed_joints
        if joints_local_rot is not None:
            joints_local_rot = to_torch(joints_local_rot, device=self.joints_local_rot.device).to(
                dtype=self.joints_local_rot.dtype
            )
            self.joints_local_rot[frame_idx] = joints_local_rot
            if joints_rot is None or joints_pos is None:
                # need to update with FK
                new_posed_joints, new_global_rots = batch_rigid_transform(
                    self.joints_local_rot[frame_idx : frame_idx + 1],
                    self.skeleton.neutral_joints[None].to(self.joints_local_rot.device),
                    self.skeleton.joint_parents.to(self.joints_local_rot.device),
                    self.skeleton.root_idx,
                )
                new_posed_joints = (
                    new_posed_joints[0]
                    + self.joints_pos[frame_idx, self.skeleton.root_idx : self.skeleton.root_idx + 1]
                    - self.skeleton.neutral_joints[[self.skeleton.root_idx]]
                )
                if joints_rot is None:
                    self.joints_rot[frame_idx] = new_global_rots[0]
                if joints_pos is None:
                    self.joints_pos[frame_idx] = new_posed_joints
        if foot_contacts is not None:
            foot_contacts = to_torch(foot_contacts, device=self.foot_contacts.device).to(dtype=self.foot_contacts.dtype)
            self.foot_contacts[frame_idx] = foot_contacts

        if self.character.skeleton_mesh is not None:
            self.character.skeleton_mesh.update_mesh_info_cache(self.joints_pos[frame_idx], frame_idx)
        if self.character.skinned_mesh is not None:
            self.character.update_skinning_cache(self.joints_pos[frame_idx], self.joints_rot[frame_idx], frame_idx)

    def clear(self):
        self.character.clear()

    #
    # Editing helpers
    #
    def get_current_projected_root_pos(self) -> np.ndarray:
        """Get the projected root position on the ground at the current frame."""
        root_pos = self.joints_pos[self.cur_frame_idx, self.skeleton.root_idx].clone()
        root_pos[1] = 0.0
        return to_numpy(root_pos)

    def get_projected_root_pos(self, start_frame_idx: int, end_frame_idx: int = None) -> np.ndarray:
        """If requested frames are out of range, simply pads with the last frame to get expected
        length."""
        if end_frame_idx is None:
            expected_len = 1
        else:
            expected_len = end_frame_idx - start_frame_idx + 1
        if start_frame_idx >= self.length:
            start_frame_idx = self.length - 1
        if end_frame_idx is None or expected_len == 1:
            root_pos = self.joints_pos[start_frame_idx, self.skeleton.root_idx].clone()
            root_pos[1] = 0.0
            return to_numpy(root_pos)
        else:
            if end_frame_idx >= self.length:
                end_frame_idx = self.length = 1
            root_pos = self.joints_pos[start_frame_idx : end_frame_idx + 1, self.skeleton.root_idx].clone()
            root_pos[:, 1] = 0.0
            if root_pos.shape[0] < expected_len:
                # pad with the last root position
                root_pos = torch.cat(
                    [
                        root_pos,
                        root_pos[-1:].repeat(expected_len - root_pos.shape[0], 1),
                    ],
                    dim=0,
                )
            return to_numpy(root_pos)

    def set_projected_root_pos_path(
        self,
        root_pos_path: np.ndarray | torch.Tensor,
        min_frame_idx: int = None,
        max_frame_idx: int = None,
    ):
        """Sets the projected root position path for the character motion.

        Can set only a subset of the path by providing min_frame_idx and max_frame_idx. If not provided, will set the full
        path.

        Args:
            root_pos_path: torch.Tensor, [T, 2] projected root positions
            min_frame_idx: int, optional, minimum frame index to set the path at
            max_frame_idx: int, optional, maximum frame index to set the path at
        """
        if min_frame_idx is not None or max_frame_idx is not None:
            assert min_frame_idx is not None and max_frame_idx is not None, (
                "min_frame_idx and max_frame_idx must be provided if setting path at specific frames"
            )
            if min_frame_idx >= self.length:
                # both are out of bounds
                return
            max_frame_idx = min(max_frame_idx, self.length - 1)
            root_pos_path = root_pos_path[min_frame_idx : max_frame_idx + 1]
        else:
            assert root_pos_path.shape[0] == self.length
            min_frame_idx = 0
            max_frame_idx = self.length - 1

        cur_joints_pos = self.joints_pos.clone()[min_frame_idx : max_frame_idx + 1]
        root_pos_tensor = to_torch(root_pos_path, device=cur_joints_pos.device, dtype=cur_joints_pos.dtype)
        diff = root_pos_tensor - cur_joints_pos[:, self.skeleton.root_idx, [0, 2]]
        cur_joints_pos[:, :, [0, 2]] += diff.unsqueeze(1)
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            rel_idx = frame_idx - min_frame_idx
            self.update_pose_at_frame(
                frame_idx,
                joints_pos=cur_joints_pos[rel_idx],
                joints_rot=self.joints_rot[frame_idx],
                joints_local_rot=self.joints_local_rot[frame_idx],
            )
        # update immediately to show changes
        self.set_frame(self.cur_frame_idx)

    def get_joints_pos(self, start_frame_idx: int, end_frame_idx: int = None) -> np.ndarray:
        """If requested frames are out of range, simply pads with the last frame to get expected
        length."""
        if end_frame_idx is None:
            expected_len = 1
        else:
            expected_len = end_frame_idx - start_frame_idx + 1
        if start_frame_idx >= self.length:
            start_frame_idx = self.length - 1
        if end_frame_idx is None or expected_len == 1:
            return to_numpy(self.joints_pos[start_frame_idx].clone())
        else:
            if end_frame_idx >= self.length:
                end_frame_idx = self.length - 1
            return_joints_pos = self.joints_pos[start_frame_idx : end_frame_idx + 1].clone()
            if return_joints_pos.shape[0] < expected_len:
                # pad with the last pose
                return_joints_pos = torch.cat(
                    [
                        return_joints_pos,
                        return_joints_pos[-1:].repeat(expected_len - return_joints_pos.shape[0], 1, 1),
                    ],
                    dim=0,
                )
            return to_numpy(return_joints_pos)

    def get_joints_rot(self, start_frame_idx: int, end_frame_idx: int = None) -> np.ndarray:
        """If requested frames are out of range, simply pads with the last frame to get expected
        length."""
        if end_frame_idx is None:
            expected_len = 1
        else:
            expected_len = end_frame_idx - start_frame_idx + 1
        if start_frame_idx >= self.length:
            start_frame_idx = self.length - 1
        if end_frame_idx is None or expected_len == 1:
            return to_numpy(self.joints_rot[start_frame_idx].clone())
        else:
            if end_frame_idx >= self.length:
                end_frame_idx = self.length - 1
            return_joints_rot = self.joints_rot[start_frame_idx : end_frame_idx + 1].clone()
            if return_joints_rot.shape[0] < expected_len:
                # pad with the last pose
                return_joints_rot = torch.cat(
                    [
                        return_joints_rot,
                        return_joints_rot[-1:].repeat(expected_len - return_joints_rot.shape[0], 1, 1, 1),
                    ],
                    dim=0,
                )
            return to_numpy(return_joints_rot)

    def get_current_joints_pos(self) -> torch.Tensor:
        return self.joints_pos[self.cur_frame_idx].clone()

    def get_current_joints_rot(self) -> torch.Tensor:
        return self.joints_rot[self.cur_frame_idx].clone()

    def add_root_translation_gizmo(self, constraints: dict):
        """Create and initialize gizmo to control the root translation."""
        # TODO: could also allow rotation around y-axis
        self.root_translation_gizmo = self.server.scene.add_transform_controls(
            f"/{self.name}/gizmo_root_translation",
            scale=0.5,
            line_width=2.5,
            active_axes=(True, False, True),  # only allow translation on xz plane
            disable_axes=False,
            disable_sliders=False,
            disable_rotations=True,
            depth_test=False,  # render even when occluded
        )
        init_position = self.get_current_projected_root_pos()
        self.root_translation_gizmo.position = init_position

        @self.root_translation_gizmo.on_update
        def _(_):
            self.updating_root_translation_gizmo = True
            # translate to gizmo position
            new_root_pos = to_torch(
                self.root_translation_gizmo.position,
                device=self.joints_pos.device,
            ).to(dtype=self.joints_pos.dtype)
            cur_joints_pos = self.joints_pos[self.cur_frame_idx].clone()
            root_diff = new_root_pos - cur_joints_pos[self.skeleton.root_idx]
            root_diff[1] = 0.0  # don't change height
            cur_joints_pos += root_diff[None]
            self.update_pose_at_frame(
                self.cur_frame_idx,
                joints_pos=cur_joints_pos,
                joints_rot=self.joints_rot[self.cur_frame_idx],
                joints_local_rot=self.joints_local_rot[self.cur_frame_idx],
            )

            self.updating_root_translation_gizmo = False
            # update immediately to show user changes
            self.set_frame(self.cur_frame_idx)
            # update the 2D waypoint constraints as well if there is one
            if "2D Root" in constraints:
                root_2d_contraints = constraints["2D Root"]
                # if there is a constraint at that frame, we want to update it
                frame_idx = self.cur_frame_idx
                if frame_idx in root_2d_contraints.keyframes:
                    for keyframe_id in root_2d_contraints.frame2keyid[frame_idx]:
                        # add will modify the existing constraint
                        root_2d_contraints.add_keyframe(
                            keyframe_id,
                            frame_idx,
                            root_pos=new_root_pos,
                            exists_ok=True,
                        )
            if "Full-Body" in constraints:
                full_body_constraints = constraints["Full-Body"]
                # if there is a constraint at that frame, we want to update it
                frame_idx = self.cur_frame_idx
                if frame_idx in full_body_constraints.keyframes:
                    current_dict = full_body_constraints.keyframes[frame_idx]
                    for keyframe_id in full_body_constraints.frame2keyid[frame_idx]:
                        # add will modify the existing constraint
                        full_body_constraints.add_keyframe(
                            keyframe_id,
                            frame_idx,
                            joints_pos=cur_joints_pos,
                            joints_rot=current_dict["joints_rot"],
                            exists_ok=True,
                        )
            if "End-Effectors" in constraints:
                end_effector_constraints = constraints["End-Effectors"]
                # if there is a constraint at that frame, we want to update it
                frame_idx = self.cur_frame_idx
                if frame_idx in end_effector_constraints.keyframes:
                    current_dict = end_effector_constraints.keyframes[frame_idx]
                    for keyframe_id, _ in end_effector_constraints.frame2keyid[frame_idx]:
                        # add will modify the existing constraint
                        end_effector_constraints.add_keyframe(
                            keyframe_id,
                            frame_idx,
                            joints_pos=cur_joints_pos,
                            joints_rot=current_dict["joints_rot"],
                            joint_names=current_dict["joint_names"],
                            end_effector_type=current_dict["end_effector_type"],
                            exists_ok=True,
                        )

    def add_joint_gizmos(self, constraints: dict):
        self.joint_gizmos = []

        joint_axis_indices = None
        hidden_gizmo_joints = None
        if isinstance(self.skeleton, G1Skeleton34):
            joint_axis_indices = _get_g1_joint_axis_indices()
            hidden_gizmo_joints = set(
                self.skeleton.left_hand_joint_names
                + self.skeleton.right_hand_joint_names
                + self.skeleton.left_foot_joint_names
                + self.skeleton.right_foot_joint_names
            )
        elif isinstance(self.skeleton, CoreSkeleton27):
            hidden_gizmo_joints = {
                "RightHandThumb1",
                "RightHandEnd",
                "LeftHandThumb1",
                "LeftHandEnd",
            }

        joints_wxyzs = tf.SO3.from_matrix(self.joints_local_rot[self.cur_frame_idx].cpu().numpy()).wxyz
        for joint_idx in range(self.skeleton.nbjoints):
            disable_axes = True  # by default, only rotation controls
            disable_sliders = True
            if joint_idx == self.skeleton.root_idx:
                disable_axes = False  # allow translation for root
                disable_sliders = False
            active_axes = (True, True, True)
            if joint_axis_indices is not None:
                joint_name = self.skeleton.bone_order_names[joint_idx]
                axis_idx = joint_axis_indices.get(joint_name)
                if axis_idx is not None:
                    # PivotControls shows rotation handles when a plane is active.
                    # To allow rotation about one axis, enable the other two axes.
                    active_axes = (
                        axis_idx != 0,
                        axis_idx != 1,
                        axis_idx != 2,
                    )
            joint_visible = True
            if hidden_gizmo_joints is not None:
                joint_name = self.skeleton.bone_order_names[joint_idx]
                joint_visible = joint_name not in hidden_gizmo_joints
            cur_joint_gizmo = self.server.scene.add_transform_controls(
                f"/{self.name}/gizmo_joint_{joint_idx}",
                scale=0.075,
                line_width=4.0,
                active_axes=active_axes,
                disable_axes=disable_axes,
                disable_sliders=disable_sliders,
                disable_rotations=False,
                depth_test=False,  # render even when occluded
                position=self.joints_pos[self.cur_frame_idx, joint_idx].cpu().numpy(),
                wxyz=joints_wxyzs[joint_idx],
                visible=joint_visible,
            )
            self.joint_gizmos.append(cur_joint_gizmo)

            def set_callback_in_closure(i: int) -> None:
                @cur_joint_gizmo.on_update
                def _(_) -> None:
                    self.updating_joint_gizmos = True
                    new_local_joint_rots = self.joints_local_rot[self.cur_frame_idx].clone()
                    new_local_rot = tf.SO3(self.joint_gizmos[i].wxyz)
                    new_local_rot_mat_np = new_local_rot.as_matrix()
                    if joint_axis_indices is not None:
                        joint_name = self.skeleton.bone_order_names[i]
                        axis_idx = joint_axis_indices.get(joint_name)
                        if axis_idx is not None:
                            rotvec = new_local_rot.log()
                            axis = np.zeros(3, dtype=np.float64)
                            axis[axis_idx] = 1.0
                            new_local_rot_mat_np = tf.SO3.exp(rotvec[axis_idx] * axis).as_matrix()
                    new_local_rot_mat = torch.tensor(new_local_rot_mat_np).to(new_local_joint_rots.device)
                    new_local_joint_rots[i] = new_local_rot_mat

                    self.update_pose_at_frame(
                        self.cur_frame_idx,
                        joints_local_rot=new_local_joint_rots,
                    )

                    # handle root translation separately
                    cur_joints_pos = self.joints_pos[self.cur_frame_idx].clone()
                    if i == self.skeleton.root_idx:
                        new_root_pos = to_torch(
                            self.joint_gizmos[i].position,
                            device=self.joints_pos.device,
                        ).to(dtype=self.joints_pos.dtype)
                        root_diff = new_root_pos - self.joints_pos[self.cur_frame_idx, i]
                        if torch.norm(root_diff) > 1e-3:
                            # the root translation has been changed
                            # translate to gizmo position
                            cur_joints_pos += root_diff[None]
                            self.update_pose_at_frame(
                                self.cur_frame_idx,
                                joints_pos=cur_joints_pos,
                                joints_rot=self.joints_rot[self.cur_frame_idx],
                                joints_local_rot=self.joints_local_rot[self.cur_frame_idx],
                            )

                    self.updating_joint_gizmos = False

                    # update immediately to show user changes
                    self.set_frame(self.cur_frame_idx)

                    if i == self.skeleton.root_idx:
                        # update the 2D waypoint constraints as well if there is one
                        if "2D Root" in constraints:
                            root_2d_contraints = constraints["2D Root"]
                            # if there is a constraint at that frame, we want to update it
                            frame_idx = self.cur_frame_idx
                            if frame_idx in root_2d_contraints.keyframes:
                                new_root_pos[1] = 0.0  # force y to 0
                                for keyframe_id in root_2d_contraints.frame2keyid[frame_idx]:
                                    # add will modify the existing constraint
                                    root_2d_contraints.add_keyframe(
                                        keyframe_id,
                                        frame_idx,
                                        root_pos=new_root_pos,
                                        exists_ok=True,
                                    )

                    if "Full-Body" in constraints:
                        full_body_constraints = constraints["Full-Body"]
                        # if there is a constraint at that frame, we want to update it
                        frame_idx = self.cur_frame_idx
                        if frame_idx in full_body_constraints.keyframes:
                            for keyframe_id in full_body_constraints.frame2keyid[frame_idx]:
                                # add will modify the existing constraint
                                full_body_constraints.add_keyframe(
                                    keyframe_id,
                                    frame_idx,
                                    joints_pos=self.joints_pos[frame_idx],
                                    joints_rot=self.joints_rot[frame_idx],
                                    exists_ok=True,
                                )
                    if "End-Effectors" in constraints:
                        end_effector_constraints = constraints["End-Effectors"]
                        # if there is a constraint at that frame, we want to update it
                        frame_idx = self.cur_frame_idx
                        if frame_idx in end_effector_constraints.keyframes:
                            current_dict = end_effector_constraints.keyframes[frame_idx]
                            for keyframe_id, _ in end_effector_constraints.frame2keyid[frame_idx]:
                                # add will modify the existing constraint
                                end_effector_constraints.add_keyframe(
                                    keyframe_id,
                                    frame_idx,
                                    joints_pos=self.joints_pos[frame_idx],
                                    joints_rot=self.joints_rot[frame_idx],
                                    joint_names=current_dict["joint_names"],
                                    end_effector_type=current_dict["end_effector_type"],
                                    exists_ok=True,
                                )

            set_callback_in_closure(joint_idx)

    def clear_all_gizmos(self):
        self.updating_root_translation_gizmo = True
        self.updating_joint_gizmos = True
        if self.root_translation_gizmo is not None:
            self.server.scene.remove_by_name(self.root_translation_gizmo.name)
            self.root_translation_gizmo = None
        if self.joint_gizmos is not None:
            for joint_gizmo in self.joint_gizmos:
                self.server.scene.remove_by_name(joint_gizmo.name)
            self.joint_gizmos = None
        self.updating_root_translation_gizmo = False
        self.updating_joint_gizmos = False


#
# Constraint classes
#


class ConstraintSet:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        self.name = name
        self.server = server
        self.skeleton = skeleton
        self.display_name = display_name if display_name is not None else name

        self.keyframes = dict()  # frame_idx -> poses
        self.frame2keyid = dict()  # frame_idx -> list of keyframe ids at this frame
        self.scene_elements = dict()  # frame_idx -> meshes, labels, etc.
        self.interval_labels = dict()  # (start_frame_idx, end_frame_idx) -> interval_label
        self.labels_visible = True

    def set_label_visibility(self, visible: bool) -> None:
        """Show or hide constraint labels without deleting them."""
        self.labels_visible = visible
        for scene_data in self.scene_elements.values():
            label = scene_data.get("label")
            if label is not None:
                label.visible = visible
        for interval_label in self.interval_labels.values():
            interval_label.visible = visible

    def add_keyframe(self, keyframe_id: str, frame_idx: int, pose_data: torch.Tensor):
        """Adds a single keyframe at the given frame with the given pose data.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            pose_data: torch.Tensor, e.g. full-body pose, EE pose, 2D root pose, etc.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        pose_seq_data: torch.Tensor,
    ):
        """Adds a keyframe interval between the given start and end frames with the given pose data.

        Args:
            interval_id: str, id for the interval. Must be unique within the given start_frame_idx and end_frame_idx.
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
            pose_seq_data: torch.Tensor, data for constrained interval, e.g. full-body poses, EE poses, 2D root poses, etc.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _add_interval_label(self, start_frame_idx: int, end_frame_idx: int):
        """
        Adds an interval label between the given start and end frames
        Args:
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
        """
        mid = int((start_frame_idx + end_frame_idx) / 2)
        interval_label_pos = self._get_label_pos(mid)
        interval_label = self.server.scene.add_label(
            name=f"/{self.name}/interval_label_{start_frame_idx}_{end_frame_idx}",
            text=f"{self.display_name} @ [{start_frame_idx}, {end_frame_idx}]",
            position=interval_label_pos,
            font_size_mode="screen",
            font_screen_scale=0.7,
            anchor="center-center",
        )
        interval_label.visible = self.labels_visible
        self.interval_labels[(start_frame_idx, end_frame_idx)] = interval_label

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        """
        Removes a keyframe at the given frame
        Args:
            keyframe_id: str, id for the keyframe to remove
            frame_idx: int, frame index to remove the keyframe at
        """
        raise NotImplementedError("Subclasses must implement this method")

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        """
        Removes an interval between the given start and end frames
        Args:
            interval_id: str, id for the interval to remove
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_label_pos(self, frame_idx: int):
        """
        Returns the position of where to place the displayed label for the given frame index
        Args:
            frame_idx: int, frame index to get the label position for
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _remove_interval_and_update_label(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        """
        Removes an interval between the given start and end frames and updates the label
        Args:
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
        """
        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            self.remove_keyframe(interval_id, frame_idx)

        # Update interval labels that overlap with the removed range
        intervals_to_update = []
        for (interval_start, interval_end), label in list(self.interval_labels.items()):
            # Check if intervals overlap
            if interval_start <= end_frame_idx and interval_end >= start_frame_idx:
                intervals_to_update.append((interval_start, interval_end, label))

        for interval_start, interval_end, label in intervals_to_update:
            # Remove old label from scene and dict
            self.server.scene.remove_by_name(label.name)
            del self.interval_labels[(interval_start, interval_end)]

            new_start, new_end = update_interval(interval_start, interval_end, start_frame_idx, end_frame_idx)

            if new_start is None or new_end is None:
                continue

            # Create updated label with new range
            if new_start <= new_end:
                # Position label at midpoint - these keyframes are guaranteed to exist
                # since the new range is outside the removal range
                mid_frame = (new_start + new_end) // 2
                label_pos = self._get_label_pos(mid_frame)
                new_label = self.server.scene.add_label(
                    name=f"/{self.name}/interval_label_{new_start}_{new_end}",
                    text=f"{self.display_name} @ [{new_start}, {new_end}]",
                    position=label_pos,
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="center-center",
                )
                new_label.visible = self.labels_visible
                self.interval_labels[(new_start, new_end)] = new_label

    def get_constraint_info(self, device: Optional[str] = None):
        """Returns constraint information for generation (torch) or UI (numpy)."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_frame_idx(self):
        """Returns all constrained frame indices in the set."""
        return [frame_idx for frame_idx in list(self.keyframes.keys())]

    def set_keyframe_visibility(self, keyframe_idx: int, visible: bool):
        """Sets the visibility of scene elements at the given keyframe index.

        Args:
            keyframe_idx: int, keyframe index to set visibility for
            visible: bool, whether to make the elements visible
        """
        raise NotImplementedError("Subclasses must implement this method")

    def clear(self, frame_idx: Optional[int] = None):
        """
        Clears all keyframes and intervals from the constraint set
        Args:
            frame_idx: int, sing frame index to clear if given
        """
        raise NotImplementedError("Subclasses must implement this method")


def build_constraint_set_table_markdown(constraint_list: List[ConstraintSet]):
    markdown = "| Track | Frame Num |\n"
    markdown += "|------|----------|\n"

    # Sort constraints by frame_idx
    for constraint in constraint_list:
        frame_info = constraint.get_frame_idx()
        if len(frame_info) > 0:
            frame_info = ", ".join([str(frame) for frame in sorted(frame_info)])
        else:
            frame_info = "-"
        markdown += f"| {constraint.display_name} | {frame_info} |\n"

    return markdown


def update_interval(interval_start, interval_end, start_frame_idx, end_frame_idx):
    """Updates an interval after removing the range from start_frame_idx to end_frame_idx."""
    # Calculate new range after removing [start_frame_idx, end_frame_idx]
    # Case 1: Removal fully contains the interval -> delete entirely
    if start_frame_idx <= interval_start and end_frame_idx >= interval_end:
        return None, None  # Already removed, don't recreate
    # Case 2: Removal is at the start of interval -> shrink from start
    elif start_frame_idx <= interval_start and end_frame_idx < interval_end:
        new_start = end_frame_idx + 1
        new_end = interval_end
    # Case 3: Removal is at the end of interval -> shrink from end
    elif start_frame_idx > interval_start and end_frame_idx >= interval_end:
        new_start = interval_start
        new_end = start_frame_idx - 1
    # Case 4: Removal is in the middle -> keep the larger portion
    else:  # start_frame_idx > interval_start and end_frame_idx < interval_end
        left_size = start_frame_idx - interval_start
        right_size = interval_end - end_frame_idx
        if left_size >= right_size:
            new_start = interval_start
            new_end = start_frame_idx - 1
        else:
            new_start = end_frame_idx + 1
            new_end = interval_end
    return new_start, new_end


class FullbodyKeyframeSet(ConstraintSet):
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        super().__init__(name, server, skeleton, display_name=display_name)

    def add_keyframe(
        self,
        keyframe_id: str,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        viz_label: bool = True,
        exists_ok: bool = False,
    ):
        """Adds a single full-body keyframe at the given frame or updates the existing one at this
        frame. Note if a keyframe already exists at this frame, it will be updated to the given
        pose.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            joints_pos: torch.Tensor, [J, 3] joints positions to add the keyframe at
        """
        # create/update scene elements
        if frame_idx in self.keyframes:
            skeleton_mesh = self.scene_elements[frame_idx]["skeleton_mesh"]
            skeleton_mesh.set_pose(to_torch(joints_pos))
            if viz_label and "label" in self.scene_elements[frame_idx]:
                label = self.scene_elements[frame_idx]["label"]
                label.position = to_numpy(joints_pos)[self.skeleton.root_idx]
                label.visible = self.labels_visible
        else:
            # create skeleton to visualize the full-body constraint
            skeleton_mesh = SkeletonMesh(
                f"/{self.name}/skeleton_{frame_idx}",
                self.server,
                self.skeleton,
                joint_color=(255, 235, 0),
                bone_color=(255, 0, 0),
                starting_joints_pos=to_torch(joints_pos),
            )
            self.scene_elements[frame_idx] = {
                "skeleton_mesh": skeleton_mesh,
            }
            if viz_label:
                label = self.server.scene.add_label(
                    name=f"/{self.name}/label_{frame_idx}",
                    text=f"{self.display_name} @ {frame_idx}",
                    position=to_numpy(joints_pos)[self.skeleton.root_idx],
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="center-center",
                )
                label.visible = self.labels_visible
                self.scene_elements[frame_idx]["label"] = label

        # set/update data
        self.keyframes[frame_idx] = {
            "joints_pos": to_numpy(joints_pos),
            "joints_rot": to_numpy(joints_rot),
        }

        if frame_idx not in self.frame2keyid:
            self.frame2keyid[frame_idx] = []

        if keyframe_id in self.frame2keyid[frame_idx]:
            if not exists_ok:
                raise AssertionError("keyframe_id already exists in this frame!")
        else:
            self.frame2keyid[frame_idx].append(keyframe_id)

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
    ):
        """Adds a full-body keyframe interval between the given start and end frames.

        Args:
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
            joints_pos: torch.Tensor, [T, J, 3] joints positions within the interval
        """
        assert joints_pos.shape[0] == end_frame_idx - start_frame_idx + 1
        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            rel_idx = frame_idx - start_frame_idx
            self.add_keyframe(
                interval_id,
                frame_idx,
                joints_pos[rel_idx],
                joints_rot[rel_idx],
                viz_label=False,
            )

        # add separate interval label
        self._add_interval_label(start_frame_idx, end_frame_idx)

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        if frame_idx not in self.keyframes:
            return
        if keyframe_id not in self.frame2keyid[frame_idx]:
            return
        self.frame2keyid[frame_idx].remove(keyframe_id)
        if len(self.frame2keyid[frame_idx]) == 0:
            del self.frame2keyid[frame_idx]
            self.clear(frame_idx)

    def _get_label_pos(self, frame_idx: int):
        return self.keyframes[frame_idx]["joints_pos"][self.skeleton.root_idx]

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        self._remove_interval_and_update_label(interval_id, start_frame_idx, end_frame_idx)

    def get_constraint_info(self, device: Optional[str] = None):
        all_joints_pos = []
        all_joints_rot = []
        for v in self.keyframes.values():
            joints_pos = to_torch(v["joints_pos"], device=device)
            joints_rot = to_torch(v["joints_rot"], device=device)
            if len(joints_pos.shape) == 2:
                all_joints_pos.append(joints_pos[None])
            else:
                all_joints_pos.append(joints_pos)
            if len(joints_rot.shape) == 3:
                all_joints_rot.append(joints_rot[None])
            else:
                all_joints_rot.append(joints_rot)

        all_joints_pos = torch.cat(all_joints_pos, dim=0) if len(all_joints_pos) > 0 else None
        all_joints_rot = torch.cat(all_joints_rot, dim=0) if len(all_joints_rot) > 0 else None

        return {
            "frame_idx": self.get_frame_idx(),
            "joints_pos": all_joints_pos,
            "joints_rot": all_joints_rot,
        }

    def set_keyframe_visibility(self, keyframe_idx: int, visible: bool):
        """Sets the visibility of scene elements at the given keyframe index."""
        if keyframe_idx not in self.scene_elements:
            return

        scene_elements = self.scene_elements[keyframe_idx]
        if "skeleton_mesh" in scene_elements:
            skeleton_mesh = scene_elements["skeleton_mesh"]
            if hasattr(skeleton_mesh, "joints_batched_mesh"):
                skeleton_mesh.joints_batched_mesh.visible = visible
            if hasattr(skeleton_mesh, "bones_batched_mesh"):
                skeleton_mesh.bones_batched_mesh.visible = visible

        if "label" in scene_elements:
            label = scene_elements["label"]
            if hasattr(label, "visible"):
                label.visible = visible

    def clear(self, frame_idx: Optional[int] = None):
        frame_idx_list = list(self.keyframes.keys()) if frame_idx is None else [frame_idx]
        for fidx in frame_idx_list:
            self.scene_elements[fidx]["skeleton_mesh"].clear()
            if "ee_rotation_axes" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["ee_rotation_axes"].name)
            if "label" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["label"].name)

            self.keyframes.pop(fidx)
            self.scene_elements.pop(fidx)

        if frame_idx is None:
            # clear all interval labels if clearing all keyframes
            for interval_label in list(self.interval_labels.values()):
                self.server.scene.remove_by_name(interval_label.name)
            self.interval_labels.clear()


class EEJointsKeyframeSet(ConstraintSet):
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        super().__init__(name, server, skeleton, display_name=display_name)

        # frame_idx -> list of (keyframe_id, joint_names) at this frame
        self.frame2keyid = dict()

    def create_scene_elements(
        self,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: Optional[torch.Tensor | np.ndarray],
        joint_names: List[str],
        viz_label: bool = True,
    ):
        # create skeleton to visualize the full-body constraint
        ee_joint_indices = []
        ee_gizmo_indices = []
        constrained_bone_idx = []
        for joint_name in joint_names:
            if joint_name == "Hips":
                continue
            elif joint_name in ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]:
                expanded_joint_names = {
                    "LeftHand": self.skeleton.left_hand_joint_names,
                    "RightHand": self.skeleton.right_hand_joint_names,
                    "LeftFoot": self.skeleton.left_foot_joint_names,
                    "RightFoot": self.skeleton.right_foot_joint_names,
                }[joint_name]
                ee_joint_indices.extend([self.skeleton.bone_order_names_index[joint] for joint in expanded_joint_names])
                if len(expanded_joint_names) > 1:
                    ee_gizmo_indices.extend(
                        [self.skeleton.bone_order_names_index[joint] for joint in expanded_joint_names[:-1]]
                    )
                constrained_bone_idx.extend(
                    [self.skeleton.bone_order_names_index[joint] - 1 for joint in expanded_joint_names[1:]]
                )
            else:
                raise ValueError(f"Invalid joint name: {joint_name}")

        # de-duplicate while preserving order
        ee_joint_indices = list(dict.fromkeys(ee_joint_indices))
        ee_gizmo_indices = list(dict.fromkeys(ee_gizmo_indices))
        constrained_bone_idx = list(dict.fromkeys(constrained_bone_idx))

        constrained_idx = np.array([self.skeleton.root_idx] + ee_joint_indices, dtype=np.intp)
        constrained_bone_idx = np.array(constrained_bone_idx, dtype=np.intp)

        # create skeleton to visualize the full-body constraint
        joint_color = np.full((self.skeleton.nbjoints, 3), (220, 220, 220))
        bone_color = np.full((self.skeleton.nbjoints - 1, 3), (220, 220, 220))
        # color constrained joints and bones red
        joint_color[constrained_idx] = (255, 0, 0)
        if len(constrained_bone_idx) > 0:
            bone_color[constrained_bone_idx] = (255, 0, 0)
        skeleton_mesh = SkeletonMesh(
            f"/{self.name}/skeleton_{frame_idx}",
            self.server,
            self.skeleton,
            joint_color=joint_color,
            bone_color=bone_color,
            starting_joints_pos=to_torch(joints_pos),
        )

        self.scene_elements[frame_idx] = {
            "skeleton_mesh": skeleton_mesh,
        }
        joints_pos_np = to_numpy(joints_pos)
        joints_rot_np = to_numpy(joints_rot) if joints_rot is not None else None
        if joints_rot_np is not None and len(ee_gizmo_indices) > 0:
            ee_axes = self.server.scene.add_batched_axes(
                f"/{self.name}/ee_rot_axes_{frame_idx}",
                batched_wxyzs=tf.SO3.from_matrix(joints_rot_np[ee_gizmo_indices]).wxyz,
                batched_positions=joints_pos_np[ee_gizmo_indices],
                axes_length=0.07,
                axes_radius=0.007,
            )
            self.scene_elements[frame_idx]["ee_rotation_axes"] = ee_axes
        if viz_label:
            label = self.server.scene.add_label(
                name=f"/{self.name}/label_{frame_idx}",
                text=f"{self.display_name} @ {frame_idx}",
                position=joints_pos_np[self.skeleton.root_idx] + np.array([0.0, 0.05, 0.0]),
                font_size_mode="screen",
                font_screen_scale=0.7,
                anchor="bottom-center",
            )
            label.visible = self.labels_visible
            self.scene_elements[frame_idx]["label"] = label

    def add_keyframe(
        self,
        keyframe_id: str,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        joint_names: List[str],
        end_effector_type: str,
        viz_label: bool = True,
        exists_ok: bool = False,
    ):
        """Adds a single EE keyframe at the given frame or updates the existing one at this frame.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            joints_pos: torch.Tensor, [J, 3] joints positions to add the keyframe at
            joints_rot: torch.Tensor, [J, 3, 3] joints rotation matrices to add the keyframe at
            joint_names: List[str], names of the joints to add the keyframe at
        """
        need_create_viz = True
        joint_names_input = joint_names

        if not isinstance(end_effector_type, set):
            end_effector_type = set([end_effector_type])

        # create/update scene elements
        if frame_idx in self.keyframes:
            if joint_names != self.keyframes[frame_idx]["joint_names"]:
                # merge together with existing constraint if needed
                joint_names = set(joint_names)
                joint_names.update(set(self.keyframes[frame_idx]["joint_names"]))
                joint_names = list(joint_names)
                end_effector_type.update(self.keyframes[frame_idx]["end_effector_type"])
                # need to re-create viz elements
                self.clear(frame_idx)
            else:
                need_create_viz = False
                # overwrite the pose with the latest one
                skeleton_mesh = self.scene_elements[frame_idx]["skeleton_mesh"]
                skeleton_mesh.set_pose(to_torch(joints_pos))
                if "ee_rotation_axes" in self.scene_elements[frame_idx]:
                    ee_gizmo_indices = []
                    for joint_name in joint_names:
                        if joint_name == "Hips":
                            continue
                        elif joint_name in [
                            "LeftHand",
                            "RightHand",
                            "LeftFoot",
                            "RightFoot",
                        ]:
                            expanded_joint_names = {
                                "LeftHand": self.skeleton.left_hand_joint_names,
                                "RightHand": self.skeleton.right_hand_joint_names,
                                "LeftFoot": self.skeleton.left_foot_joint_names,
                                "RightFoot": self.skeleton.right_foot_joint_names,
                            }[joint_name]
                            if len(expanded_joint_names) > 1:
                                ee_gizmo_indices.extend(
                                    [self.skeleton.bone_order_names_index[joint] for joint in expanded_joint_names[:-1]]
                                )
                        else:
                            raise ValueError(f"Invalid joint name: {joint_name}")
                    ee_gizmo_indices = list(dict.fromkeys(ee_gizmo_indices))
                    if len(ee_gizmo_indices) > 0:
                        ee_axes = self.scene_elements[frame_idx]["ee_rotation_axes"]
                        joints_pos_np = to_numpy(joints_pos)
                        joints_rot_np = to_numpy(joints_rot)
                        ee_axes.batched_positions = joints_pos_np[ee_gizmo_indices]
                        ee_axes.batched_wxyzs = tf.SO3.from_matrix(joints_rot_np[ee_gizmo_indices]).wxyz
                if viz_label and "label" in self.scene_elements[frame_idx]:
                    label = self.scene_elements[frame_idx]["label"]
                    label.position = to_numpy(joints_pos)[self.skeleton.root_idx]
                    label.visible = self.labels_visible

        if need_create_viz:
            self.create_scene_elements(frame_idx, joints_pos, joints_rot, joint_names, viz_label=viz_label)

        # set/update data
        self.keyframes[frame_idx] = {
            "joints_pos": to_numpy(joints_pos),
            "joints_rot": to_numpy(joints_rot),
            "joint_names": joint_names,
            "end_effector_type": end_effector_type,
        }

        if frame_idx not in self.frame2keyid:
            self.frame2keyid[frame_idx] = []

        known_keyframe_ids = {k: idx for idx, (k, _) in enumerate(self.frame2keyid[frame_idx])}

        if keyframe_id in known_keyframe_ids.keys():
            if not exists_ok:
                raise AssertionError("keyframe_id already exists in this frame!")
            idx = known_keyframe_ids[keyframe_id]
            # override previous exisiting keyframe
            self.frame2keyid[frame_idx][idx] = (keyframe_id, joint_names_input)
        else:
            # track which subset of joints are constrained by this keyframe_id
            self.frame2keyid[frame_idx].append((keyframe_id, joint_names_input))

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        joint_names: List[str],
        end_effector_type: str,
    ):
        """Adds an interval of EE keyframes at the given frame or updates the existing one at this
        frame.

        Args:
            interval_id: str, id for the interval. Must be unique within the given start_frame_idx and end_frame_idx.
            start_frame_idx: int, start frame index to add the interval at
            end_frame_idx: int, end frame index to add the interval at
            joints_pos: torch.Tensor, [T, J, 3] joints positions to add the interval at
            joints_rot: torch.Tensor, [T, J, 3, 3] joints rotation matrices to add the interval at
            joint_names: List[str], names of the joints to add for the entire interval
        """
        num_frames = end_frame_idx - start_frame_idx + 1
        joints_pos_np = to_numpy(joints_pos)
        joints_rot_np = to_numpy(joints_rot)
        assert joints_pos_np.shape[0] == num_frames
        assert joints_rot_np.shape[0] == num_frames

        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            rel_idx = frame_idx - start_frame_idx
            self.add_keyframe(
                interval_id,
                frame_idx,
                joints_pos_np[rel_idx],
                joints_rot_np[rel_idx],
                joint_names,
                end_effector_type,
                viz_label=False,
            )
        self._add_interval_label(start_frame_idx, end_frame_idx)

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        """Removes a keyframe at the given frame or updates the existing one at this frame by
        removing the specified joints.

        Args:
            keyframe_id: str, id for the keyframe to remove. This determines which joints to remove.
            frame_idx: int, frame index to remove the keyframe at
        """
        if frame_idx not in self.keyframes:
            return

        remaining_joint_names = set()
        delete_idx = None
        for i, (keyid, joint_names) in enumerate(self.frame2keyid[frame_idx]):
            if keyid == keyframe_id:
                delete_idx = i
            else:
                remaining_joint_names.update(joint_names)
        if delete_idx is None:
            # this keyframe_id is not in the specified frame
            return

        self.frame2keyid[frame_idx].pop(delete_idx)
        if len(remaining_joint_names) == 0:
            # no more keyframes in this frame, clear the frame
            del self.frame2keyid[frame_idx]
            self.clear(frame_idx)
            return

        # only deleting part of keyframe (potentially some subset of joints)
        # delete the old visualization and add a new one with the updated joint set
        new_joint_names = list(remaining_joint_names)
        self.clear(frame_idx, scene_elements_only=True)
        joints_pos = self.keyframes[frame_idx]["joints_pos"]
        joints_rot = self.keyframes[frame_idx]["joints_rot"]
        self.create_scene_elements(frame_idx, joints_pos, joints_rot, new_joint_names)
        self.keyframes[frame_idx]["joint_names"] = new_joint_names

    def _get_label_pos(self, frame_idx: int):
        return self.keyframes[frame_idx]["joints_pos"][self.skeleton.root_idx]

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        self._remove_interval_and_update_label(interval_id, start_frame_idx, end_frame_idx)

    def get_constraint_info(self, device: Optional[str] = None):
        all_joints_pos = []
        all_joints_rot = []
        all_joints_names = []
        all_end_effector_type = []
        for v in self.keyframes.values():
            joints_pos = to_torch(v["joints_pos"], device=device)
            joints_rot = to_torch(v["joints_rot"], device=device)
            if len(joints_pos.shape) == 2:
                all_joints_pos.append(joints_pos[None])
            else:
                all_joints_pos.append(joints_pos)
            if len(joints_rot.shape) == 3:
                all_joints_rot.append(joints_rot[None])
            else:
                all_joints_rot.append(joints_rot)
            all_joints_names.append(v["joint_names"])
            all_end_effector_type.append(v["end_effector_type"])

        all_joints_pos = torch.cat(all_joints_pos, dim=0) if len(all_joints_pos) > 0 else None
        all_joints_rot = torch.cat(all_joints_rot, dim=0) if len(all_joints_rot) > 0 else None

        return {
            "frame_idx": self.get_frame_idx(),
            "joints_pos": all_joints_pos,
            "joints_rot": all_joints_rot,
            "joint_names": all_joints_names,
            "end_effector_type": all_end_effector_type,
        }

    def set_keyframe_visibility(self, keyframe_idx: int, visible: bool, show_rotation_axes: bool = True):
        """Sets the visibility of scene elements at the given keyframe index."""
        if keyframe_idx not in self.scene_elements:
            return

        scene_elements = self.scene_elements[keyframe_idx]
        if "skeleton_mesh" in scene_elements:
            skeleton_mesh = scene_elements["skeleton_mesh"]
            if hasattr(skeleton_mesh, "joints_batched_mesh"):
                skeleton_mesh.joints_batched_mesh.visible = visible
            if hasattr(skeleton_mesh, "bones_batched_mesh"):
                skeleton_mesh.bones_batched_mesh.visible = visible

        if "ee_rotation_axes" in scene_elements:
            ee_axes = scene_elements["ee_rotation_axes"]
            if hasattr(ee_axes, "visible"):
                ee_axes.visible = visible and show_rotation_axes

        if "label" in scene_elements:
            label = scene_elements["label"]
            if hasattr(label, "visible"):
                label.visible = visible

    def clear(self, frame_idx: Optional[int] = None, scene_elements_only: bool = False):
        frame_idx_list = list(self.keyframes.keys()) if frame_idx is None else [frame_idx]
        for fidx in frame_idx_list:
            self.scene_elements[fidx]["skeleton_mesh"].clear()
            if "ee_rotation_axes" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["ee_rotation_axes"].name)
            if "label" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["label"].name)
            self.scene_elements.pop(fidx)
            if not scene_elements_only:
                self.keyframes.pop(fidx)

        if frame_idx is None:
            # clear all interval labels if clearing all keyframes
            for interval_label in list(self.interval_labels.values()):
                self.server.scene.remove_by_name(interval_label.name)
            self.interval_labels.clear()


class RootKeyframe2DSet(ConstraintSet):
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        super().__init__(name, server, skeleton, display_name=display_name)
        self.dense_path = False
        self.smooth_path = True
        self.line_segments = None  # visualization of dense path
        # Cache for interpolated path
        self._cached_t = None
        self._cached_path3d = None
        # Store root headings for each keyframe
        self.root_headings = {}  # frame_idx -> heading (float)

    def add_keyframe(
        self,
        keyframe_id: str,
        frame_idx: int,
        root_pos: torch.Tensor | np.ndarray,
        global_root_heading: Optional[float] = None,
        viz_label: bool = True,
        update_path: bool = True,
        exists_ok: bool = False,
        add_annulus: bool = True,
    ):
        """Adds a single 2D root keyframe at the given frame or updates the existing one at this
        frame.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            root_pos: torch.Tensor, [3] root position to add the keyframe at, y entry (index 1) should be 0
            viz_label: bool, whether to visualize the label for the keyframe
        """
        root_pos_np = to_numpy(root_pos)

        # Convert heading angle to 2D direction vector for visualization
        heading_2d = None
        if global_root_heading is not None:
            # Heading is in radians, convert to [x, z] direction vector as [cos, sin]
            heading_2d = np.array([np.cos(global_root_heading), np.sin(global_root_heading)])

        if frame_idx in self.keyframes:
            waypoint = self.scene_elements[frame_idx]["waypoint"]
            waypoint.update_position(root_pos_np, heading=heading_2d)
            if viz_label and "label" in self.scene_elements[frame_idx]:
                label = self.scene_elements[frame_idx]["label"]
                label.position = root_pos.cpu().numpy()
                label.visible = self.labels_visible
        else:
            waypoint = WaypointMesh(
                f"/{self.name}/{keyframe_id}",
                self.server,
                position=root_pos_np,
                heading=heading_2d,
                add_annulus=add_annulus,
            )
            self.scene_elements[frame_idx] = {
                "waypoint": waypoint,
            }
            if viz_label:
                label = self.server.scene.add_label(
                    name=f"/{self.name}/label_{frame_idx}",
                    text=f"{self.display_name} @ {frame_idx}",
                    position=root_pos_np,
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="bottom-left",
                )
                label.visible = self.labels_visible
                self.scene_elements[frame_idx]["label"] = label

        # set/update data
        self.keyframes[frame_idx] = root_pos_np
        if global_root_heading is not None:
            self.root_headings[frame_idx] = global_root_heading
        if frame_idx not in self.frame2keyid:
            self.frame2keyid[frame_idx] = []

        if keyframe_id in self.frame2keyid[frame_idx]:
            if not exists_ok:
                raise AssertionError("keyframe_id already exists in this frame!")
        else:
            self.frame2keyid[frame_idx].append(keyframe_id)

        # need to update path visualization
        if self.dense_path and self.line_segments is None:
            # visualize dense path with line segments
            self.line_segments = self.server.scene.add_line_segments(
                name=f"/{self.name}/line_segments",
                points=np.zeros((1, 2, 3)),
                colors=(255, 0, 0),
                line_width=5.0,
            )
        if update_path:
            self.update_line_segments()

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        root_pos: torch.Tensor | np.ndarray,
        add_annulus: bool = True,
    ):
        """Adds an interval of 2D root keyframes between the given start and end frames.

        Args:
            interval_id: str, id for the interval. Must be unique within the given start_frame_idx and end_frame_idx.
            start_frame_idx: int, start frame index to add the interval at
            end_frame_idx: int, end frame index to add the interval at
            root_pos: torch.Tensor, [T, 3] root positions to add the interval at
        """
        root_pos_np = to_numpy(root_pos)
        assert root_pos_np.shape[0] == end_frame_idx - start_frame_idx + 1
        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            rel_idx = frame_idx - start_frame_idx
            self.add_keyframe(
                f"{interval_id}/{frame_idx}",
                frame_idx,
                root_pos_np[rel_idx],
                viz_label=False,
                update_path=False,
                add_annulus=add_annulus,
            )
        self._add_interval_label(start_frame_idx, end_frame_idx)
        if self.line_segments is not None:
            self.update_line_segments()

    def set_smooth_path(self, smooth_path: bool):
        self.smooth_path = smooth_path
        if self.line_segments is not None:
            self.update_line_segments()

    def set_dense_path(self, dense_path: bool):
        """If dense_path is True, will make the path dense by interpolated between added keyframes.

        Args:
            dense_path: bool, whether to make the path dense
        """
        self.dense_path = dense_path
        if self.dense_path:
            # visualize dense path with line segments
            self.line_segments = self.server.scene.add_line_segments(
                name=f"/{self.name}/line_segments",
                points=np.zeros((1, 2, 3)),
                colors=(255, 0, 0),
                line_width=5.0,
            )
            self.update_line_segments()
        else:
            if self.line_segments is not None:
                self.server.scene.remove_by_name(self.line_segments.name)
                self.line_segments = None

    # TODO: accelerate interp1d. Note that np and torch interp gives different results, much less smooth
    def interpolate_path(self, t: np.ndarray):
        """Interpolates the path between the given frame indices.

        Args:
            t: np.ndarray, frame indices to interpolate at
        """
        cur_info = self._get_sparse_constraint_info()
        frame_idx = cur_info["frame_idx"]
        all_root_pos = cur_info["root_pos"]

        x = all_root_pos[:, 0]
        z = all_root_pos[:, 2]

        kind = "linear"
        # if self.smooth_path and len(frame_idx) >= 3:
        # kind = "quadratic"

        interp_x = interp1d(frame_idx, x, kind=kind)
        interp_z = interp1d(frame_idx, z, kind=kind)

        x_new = interp_x(t)
        z_new = interp_z(t)

        path3d = np.stack([x_new, np.zeros_like(x_new), z_new], axis=1)

        if self.smooth_path and len(frame_idx) >= 3:
            start_time = time.time()
            path3d = get_smooth_root_pos(torch.from_numpy(path3d[None]))[0].numpy()
            print(f"Time taken to smooth path: {time.time() - start_time} seconds")
        return path3d

    def update_line_segments(self, frame_idx: int = 0):
        if len(self.keyframes) < 2:
            return

        t = np.array(sorted(self.get_frame_idx()))
        if self.smooth_path:
            # more points for smoothed curve
            # t = np.linspace(t[0], t[-1], 100)
            t = np.arange(max(t[0], frame_idx), t[-1] + 1)

        path3d = self.interpolate_path(t)

        # Cache the computed t and path3d for use in get_constraint_info
        self._cached_t = t
        self._cached_path3d = path3d

        points = np.zeros((len(path3d) - 1, 2, 3))
        points[:, 0] = path3d[:-1]
        points[:, 1] = path3d[1:]

        self.line_segments.points = points

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        if frame_idx not in self.keyframes:
            return
        if keyframe_id not in self.frame2keyid[frame_idx]:
            return
        self.frame2keyid[frame_idx].remove(keyframe_id)
        if len(self.frame2keyid[frame_idx]) == 0:
            del self.frame2keyid[frame_idx]
            # Also remove heading if it exists
            if frame_idx in self.root_headings:
                del self.root_headings[frame_idx]
            self.clear(frame_idx)
            if self.line_segments is not None:
                self.update_line_segments()

    def _get_label_pos(self, frame_idx: int):
        return self.keyframes[frame_idx]

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        self._remove_interval_and_update_label(interval_id, start_frame_idx, end_frame_idx)

    def _get_sparse_constraint_info(self):
        all_root_pos = []
        all_root_headings = []
        frame_indices = self.get_frame_idx()

        for frame_idx in frame_indices:
            v = self.keyframes[frame_idx]
            # Handle both numpy arrays and torch tensors
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 1:
                    all_root_pos.append(v.unsqueeze(0).cpu().numpy())
                else:
                    all_root_pos.append(v.cpu().numpy())
            else:  # numpy array
                if len(v.shape) == 1:
                    all_root_pos.append(v[np.newaxis, :])
                else:
                    all_root_pos.append(v)
            # Get heading if it exists
            if frame_idx in self.root_headings:
                all_root_headings.append(self.root_headings[frame_idx])

        if len(all_root_pos) > 0:
            all_root_pos = np.concatenate(all_root_pos, axis=0)
        else:
            all_root_pos = None

        result = {
            "frame_idx": frame_indices,
            "root_pos": all_root_pos,
        }

        if len(all_root_headings) > 0 and len(all_root_headings) == len(frame_indices):
            result["global_root_heading"] = torch.tensor(all_root_headings, dtype=torch.float32)

        return result

    def get_constraint_info(self):
        if not self.dense_path or len(self.keyframes) == 0:
            return self._get_sparse_constraint_info()
        else:
            # Use cached path if available, otherwise compute it
            if self._cached_t is not None and self._cached_path3d is not None:
                t = self._cached_t
                path3d = self._cached_path3d
            else:
                frame_idx_list = self.get_frame_idx()
                min_frame_idx = min(frame_idx_list)
                max_frame_idx = max(frame_idx_list)
                t = np.arange(min_frame_idx, max_frame_idx + 1)
                path3d = self.interpolate_path(t)

            result = {
                "frame_idx": t.tolist(),
                "root_pos": torch.tensor(path3d, dtype=torch.float32),
            }

            # Interpolate headings if any exist
            if len(self.root_headings) > 0:
                frame_idx_list = self.get_frame_idx()
                # Check if we have headings for all keyframes
                if all(fi in self.root_headings for fi in frame_idx_list):
                    # Interpolate headings for dense path
                    heading_values = [self.root_headings[fi] for fi in sorted(frame_idx_list)]
                    heading_frames = sorted(frame_idx_list)

                    # Linear interpolation of headings
                    interpolated_headings = []
                    for frame in t:
                        if frame in self.root_headings:
                            interpolated_headings.append(self.root_headings[frame])
                        else:
                            # Find surrounding keyframes and interpolate
                            left_frame = max([f for f in heading_frames if f <= frame], default=None)
                            right_frame = min([f for f in heading_frames if f >= frame], default=None)

                            if left_frame is not None and right_frame is not None and left_frame != right_frame:
                                # Linear interpolation
                                alpha = (frame - left_frame) / (right_frame - left_frame)
                                interp_heading = (1 - alpha) * self.root_headings[
                                    left_frame
                                ] + alpha * self.root_headings[right_frame]
                                interpolated_headings.append(interp_heading)
                            elif left_frame is not None:
                                interpolated_headings.append(self.root_headings[left_frame])
                            elif right_frame is not None:
                                interpolated_headings.append(self.root_headings[right_frame])

                    if len(interpolated_headings) == len(t):
                        result["global_root_heading"] = torch.tensor(interpolated_headings, dtype=torch.float32)

            return result

    def set_keyframe_visibility(self, keyframe_idx: int, visible: bool):
        """Sets the visibility of scene elements at the given keyframe index."""
        if keyframe_idx not in self.scene_elements:
            return

        scene_elements = self.scene_elements[keyframe_idx]
        if "waypoint" in scene_elements:
            waypoint = scene_elements["waypoint"]
            # WaypointMesh has sphere, annulus, arrow_base, arrow_head components
            if hasattr(waypoint, "sphere"):
                waypoint.sphere.visible = visible
            if hasattr(waypoint, "annulus") and waypoint.annulus is not None:
                waypoint.annulus.visible = visible
            if hasattr(waypoint, "arrow_base") and waypoint.arrow_base is not None:
                waypoint.arrow_base.visible = visible
            if hasattr(waypoint, "arrow_head") and waypoint.arrow_head is not None:
                waypoint.arrow_head.visible = visible

        if "label" in scene_elements:
            label = scene_elements["label"]
            if hasattr(label, "visible"):
                label.visible = visible

    def set_interval_labels_visibility(self, frame_idx: int):
        # set interval labels visibility
        for interval in self.interval_labels.keys():
            interval_label = self.interval_labels[interval]
            start_frame_idx, end_frame_idx = interval
            visibility = frame_idx <= end_frame_idx
            interval_label.visible = visibility

    def clear(self, frame_idx: Optional[int] = None):
        frame_idx_list = list(self.keyframes.keys()) if frame_idx is None else [frame_idx]
        for fidx in frame_idx_list:
            self.scene_elements[fidx]["waypoint"].clear()
            if "label" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["label"].name)

            self.keyframes.pop(fidx)
            self.scene_elements.pop(fidx)
            # Also clear heading if it exists
            if fidx in self.root_headings:
                self.root_headings.pop(fidx)

        if frame_idx is None:
            # clear all interval labels if clearing all keyframes
            for interval_label in list(self.interval_labels.values()):
                self.server.scene.remove_by_name(interval_label.name)
            self.interval_labels.clear()

            # clear line segments if turning off dense path
            if self.line_segments is not None:
                self.server.scene.remove_by_name(self.line_segments.name)
                self.line_segments = None

            # Clear all headings when clearing everything
            self.root_headings.clear()

            # Invalidate cache when clearing all keyframes
            self._cached_t = None
            self._cached_path3d = None
        else:
            # Invalidate cache when clearing a specific keyframe
            self._cached_t = None
            self._cached_path3d = None


def load_example_cases(examples_base_dir):
    example_dirs = os.listdir(examples_base_dir)
    example_names = sorted([dir for dir in example_dirs if os.path.isdir(os.path.join(examples_base_dir, dir))])
    example_dict = {name: os.path.join(examples_base_dir, name) for name in example_names}
    return example_dict
