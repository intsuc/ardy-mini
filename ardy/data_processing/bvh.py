# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional

import numpy as np


class BvhNode:
    """Abstraction for bvh node."""

    def __init__(self, value=[], parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1 :]
        raise IndexError("key {} not found".format(key))

    def __repr__(self):
        return str(" ".join(self.value))

    @property
    def name(self):
        return self.value[1]


class Bvh:
    """Abstraction for bvh."""

    def __init__(self, data: str, backend: Optional[str] = "graph"):
        """
        Args:
            data (str): bvh file content
        """
        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.backend = backend
        self.tokenize()
        if self.backend == "np":
            # cache important info for quick access later
            self.build_data_array()
        elif self.backend == "graph":
            pass
        else:
            raise ValueError(f"Unknown backend for BVH loading: {backend}")

    def build_data_array(self):
        joints = self.get_joints()
        self.joint2idx = dict()
        self.joint2channels = dict()
        cur_idx = 0
        for joint in joints:
            self.joint2idx[joint.value[1]] = cur_idx
            cur_idx += int(joint["CHANNELS"][0])
            self.joint2channels[joint.value[1]] = joint["CHANNELS"][1:]
        self.np_data_array = np.array(self.frames, dtype=np.float32)

    def tokenize(self):
        first_round = []
        accumulator = ""
        for char in self.data:
            if char not in ("\n", "\r"):
                accumulator += char
            elif accumulator:
                first_round.append(re.split("\\s+", accumulator.strip()))
                accumulator = ""
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == "{":
                node_stack.append(node)
            elif key == "}":
                node_stack.pop()
            else:
                node = BvhNode(item)
                # print("new node: ", node, "\nparent: ", node_stack[-1])
                node_stack[-1].add_child(node)
            if item[0] == "Frame" and item[1] == "Time:":
                frame_time_found = True

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)

        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter("JOINT"):
                iterate_joints(child)

        iterate_joints(next(self.root.filter("ROOT")))
        return joints

    def get_joints_names(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter("JOINT"):
                iterate_joints(child)

        iterate_joints(next(self.root.filter("ROOT")))
        return joints

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter("JOINT")]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search("ROOT", name)
        if not found:
            found = self.search("JOINT", name)
        if found:
            return found[0]
        raise LookupError("joint not found")

    def joint_offset(self, name, idx=[0, 1, 2]):
        joint = self.get_joint(name)
        offset = joint["OFFSET"]
        if len(offset) < max(idx):
            return None
        return (float(offset[idx[0]]), float(offset[idx[1]]), float(offset[idx[2]]))

    def joint_offset_rot(self, name):
        return self.joint_offset(name, idx=[3, 4, 5])

    def joint_channels(self, name):
        if self.backend == "np":
            return self.joint2channels[name]
        else:
            joint = self.get_joint(name)
            return joint["CHANNELS"][1:]

    def get_joint_channels_index(self, joint_name):
        if self.backend == "np":
            return self.joint2idx[joint_name]
        else:
            index = 0
            for joint in self.get_joints():
                if joint.value[1] == joint_name:
                    return index
                index += int(joint["CHANNELS"][0])
            raise LookupError("joint not found")

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            raise ValueError(f"Channel {channel} not found in {channels}")
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        """Get single frame data for on specific joint and one specific channel (e.g. Xrotation)."""
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        if self.backend == "np":
            return self.np_data_array[frame_index, joint_index + channel_index]
        else:
            return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        """Get single frame data for on specific joint from multiple specific channels (e.g.
        Xrotation, Yrotation, Zrotation)."""
        values = []
        joint_index = self.get_joint_channels_index(joint)
        if self.backend == "np":
            channel_idx = [self.get_joint_channel_index(joint, channel) for channel in channels]
            channel_idx = np.array(channel_idx) + joint_index
            values = self.np_data_array[frame_index, channel_idx]
        else:
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(float(self.frames[frame_index][joint_index + channel_index]))
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        """Get all frame data for one joint from multiple channels (e.g. Xrotation, Yrotation,
        Zrotation)."""
        joint_index = self.get_joint_channels_index(joint)
        if self.backend == "np":
            channel_idx = [self.get_joint_channel_index(joint, channel) for channel in channels]
            channel_idx = np.array(channel_idx) + joint_index
            all_frames = self.np_data_array[:, channel_idx]
        else:
            all_frames = []
            for frame in self.frames:
                values = []
                for channel in channels:
                    channel_index = self.get_joint_channel_index(joint, channel)
                    if channel_index == -1 and value is not None:
                        values.append(value)
                    else:
                        values.append(float(frame[joint_index + channel_index]))
                all_frames.append(values)
        return all_frames

    def frames_joints_channels(self, joint_names, channels):
        """Get all frames for all specified joints with one specified set of channels."""
        if self.backend != "np":
            raise NotImplementedError("Only np backend is supported for this function")
        joint_indices = [(joint_name, self.joint2idx[joint_name]) for joint_name in joint_names]
        data_indices = []
        for joint_name, joint_idx in joint_indices:
            channel_indices = [self.get_joint_channel_index(joint_name, channel) for channel in channels]
            data_indices.extend([joint_idx + channel_idx for channel_idx in channel_indices])
        all_frames = self.np_data_array[:, data_indices]
        all_frames = all_frames.reshape(-1, len(joint_names), len(channels))
        return all_frames

    def joint_parent(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        try:
            return int(next(self.root.filter("Frames:")).value[1])
        except StopIteration:
            raise LookupError("number of frames not found")

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter("Frame")).value[2])
        except StopIteration:
            raise LookupError("frame time not found")
