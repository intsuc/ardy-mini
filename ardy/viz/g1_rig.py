# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import numpy as np
import trimesh
import viser
import viser.transforms as tf

from ardy.skeleton import G1Skeleton34

G1_MESH_JOINT_MAP = {
    "pelvis_skel": ["pelvis.STL", "pelvis_contour_link.STL"],
    "left_hip_pitch_skel": ["left_hip_pitch_link.STL"],
    "left_hip_roll_skel": ["left_hip_roll_link.STL"],
    "left_hip_yaw_skel": ["left_hip_yaw_link.STL"],
    "left_knee_skel": ["left_knee_link.STL"],
    "left_ankle_pitch_skel": ["left_ankle_pitch_link.STL"],
    "left_ankle_roll_skel": ["left_ankle_roll_link.STL"],
    "right_hip_pitch_skel": ["right_hip_pitch_link.STL"],
    "right_hip_roll_skel": ["right_hip_roll_link.STL"],
    "right_hip_yaw_skel": ["right_hip_yaw_link.STL"],
    "right_knee_skel": ["right_knee_link.STL"],
    "right_ankle_pitch_skel": ["right_ankle_pitch_link.STL"],
    "right_ankle_roll_skel": ["right_ankle_roll_link.STL"],
    "waist_yaw_skel": ["waist_yaw_link_rev_1_0.STL", "waist_yaw_link.STL"],
    "waist_roll_skel": ["waist_roll_link_rev_1_0.STL", "waist_roll_link.STL"],
    "waist_pitch_skel": [
        "torso_link_rev_1_0.STL",
        "torso_link.STL",
        "logo_link.STL",
        "head_link.STL",
    ],
    "left_shoulder_pitch_skel": ["left_shoulder_pitch_link.STL"],
    "left_shoulder_roll_skel": ["left_shoulder_roll_link.STL"],
    "left_shoulder_yaw_skel": ["left_shoulder_yaw_link.STL"],
    "left_elbow_skel": ["left_elbow_link.STL"],
    "left_wrist_roll_skel": ["left_wrist_roll_link.STL"],
    "left_wrist_pitch_skel": ["left_wrist_pitch_link.STL"],
    "left_wrist_yaw_skel": ["left_wrist_yaw_link.STL", "left_rubber_hand.STL"],
    "right_shoulder_pitch_skel": ["right_shoulder_pitch_link.STL"],
    "right_shoulder_roll_skel": ["right_shoulder_roll_link.STL"],
    "right_shoulder_yaw_skel": ["right_shoulder_yaw_link.STL"],
    "right_elbow_skel": ["right_elbow_link.STL"],
    "right_wrist_roll_skel": ["right_wrist_roll_link.STL"],
    "right_wrist_pitch_skel": ["right_wrist_pitch_link.STL"],
    "right_wrist_yaw_skel": ["right_wrist_yaw_link.STL", "right_rubber_hand.STL"],
}


class G1MeshRig:
    _mesh_geom_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    _mesh_transform_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        skeleton: G1Skeleton34,
        mesh_dir: str,
        color: Tuple[int, int, int],
    ):
        self.server = server
        self.skeleton = skeleton
        self.mesh_dir = mesh_dir
        self.color = color
        self.mesh_handles: list[viser.SceneHandle] = []
        self.mesh_items: list[dict[str, object]] = []
        self._defer_initial_visibility = True
        self._mujoco_to_ardy = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        self._mesh_local_transforms = self._get_mesh_local_transforms()

        for joint_name, mesh_files in G1_MESH_JOINT_MAP.items():
            if joint_name not in self.skeleton.bone_index:
                continue
            joint_idx = self.skeleton.bone_index[joint_name]
            for mesh_file in mesh_files:
                mesh_path = os.path.join(self.mesh_dir, mesh_file)
                if not os.path.exists(mesh_path):
                    continue
                vertices, faces = self._get_mesh_geometry(mesh_file, mesh_path)
                if vertices is None:
                    continue
                handle = self.server.scene.add_mesh_simple(
                    f"/{name}/g1_mesh/{os.path.splitext(mesh_file)[0]}",
                    vertices=vertices,
                    faces=faces,
                    opacity=None,
                    color=self.color,
                    wireframe=False,
                    visible=not self._defer_initial_visibility,
                )
                self.mesh_handles.append(handle)
                geom_pos, geom_rot = self._mesh_local_transforms.get(
                    mesh_file,
                    (np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
                )
                self.mesh_items.append(
                    {
                        "handle": handle,
                        "joint_idx": joint_idx,
                        "geom_pos": geom_pos,
                        "geom_rot": geom_rot,
                    }
                )

        if self._defer_initial_visibility:
            for handle in self.mesh_handles:
                handle.visible = True

    def _get_mesh_geometry(self, mesh_file: str, mesh_path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        cache_key = self.mesh_dir
        cached = self._mesh_geom_cache.get(cache_key)
        if cached is not None and mesh_file in cached:
            vertices, faces = cached[mesh_file]
            return vertices.copy(), faces.copy()

        mesh = trimesh.load_mesh(mesh_path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        vertices = mesh.vertices @ self._mujoco_to_ardy.T
        faces = mesh.faces

        if cache_key not in self._mesh_geom_cache:
            self._mesh_geom_cache[cache_key] = {}
        self._mesh_geom_cache[cache_key][mesh_file] = (vertices, faces)
        return vertices.copy(), faces.copy()

    def _get_mesh_local_transforms(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        cached = self._mesh_transform_cache.get(self.mesh_dir)
        if cached is not None:
            return {mesh_file: (pos.copy(), rot.copy()) for mesh_file, (pos, rot) in cached.items()}

        xml_path = os.path.abspath(os.path.join(self.mesh_dir, "..", "..", "xml", "g1.xml"))
        if not os.path.exists(xml_path):
            return {}
        tree = ET.parse(xml_path)
        root = tree.getroot()

        mesh_file_to_mesh_name = {}
        for mesh in root.findall(".//asset/mesh"):
            mesh_name = mesh.get("name")
            mesh_file = mesh.get("file")
            if mesh_name and mesh_file:
                mesh_file_to_mesh_name[mesh_file] = mesh_name

        mesh_name_to_transform = {}
        for geom in root.findall(".//geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue
            pos = geom.get("pos")
            quat = geom.get("quat")
            if pos is None:
                geom_pos = np.zeros(3, dtype=np.float64)
            else:
                geom_pos = np.array([float(x) for x in pos.split()], dtype=np.float64)
            if quat is None:
                geom_rot = np.eye(3, dtype=np.float64)
            else:
                wxyz = np.array([float(x) for x in quat.split()], dtype=np.float64)
                geom_rot = tf.SO3(wxyz=wxyz).as_matrix()
            mesh_name_to_transform[mesh_name] = (geom_pos, geom_rot)

        mesh_file_transforms = {}
        for mesh_file, mesh_name in mesh_file_to_mesh_name.items():
            geom_pos, geom_rot = mesh_name_to_transform.get(
                mesh_name,
                (np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
            )
            geom_pos = self._mujoco_to_ardy @ geom_pos
            geom_rot = self._mujoco_to_ardy @ geom_rot @ self._mujoco_to_ardy.T
            mesh_file_transforms[mesh_file] = (geom_pos, geom_rot)

        self._mesh_transform_cache[self.mesh_dir] = {
            mesh_file: (pos.copy(), rot.copy()) for mesh_file, (pos, rot) in mesh_file_transforms.items()
        }
        return mesh_file_transforms

    def set_visibility(self, visible: bool) -> None:
        for handle in self.mesh_handles:
            handle.visible = visible

    def set_opacity(self, opacity: float) -> None:
        for handle in self.mesh_handles:
            handle.opacity = opacity

    def set_wireframe(self, wireframe: bool) -> None:
        for handle in self.mesh_handles:
            handle.wireframe = wireframe

    def set_color(self, color: Tuple[int, int, int]) -> None:
        self.color = color
        for handle in self.mesh_handles:
            handle.color = color

    def set_pose(self, joints_pos: np.ndarray, joints_rot: np.ndarray) -> None:
        for item in self.mesh_items:
            handle = item["handle"]
            joint_idx = item["joint_idx"]
            geom_pos = item["geom_pos"]
            geom_rot = item["geom_rot"]

            joint_pos = joints_pos[joint_idx]
            joint_rot = joints_rot[joint_idx]
            mesh_pos = joint_pos + joint_rot @ geom_pos
            mesh_rot = joint_rot @ geom_rot

            handle.position = mesh_pos
            handle.wxyz = tf.SO3.from_matrix(mesh_rot).wxyz

    def clear(self) -> None:
        for handle in self.mesh_handles:
            self.server.scene.remove_by_name(handle.name)
