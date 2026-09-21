"""Hardware-free remote policy client and action adapter."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import pickle
import socket
import struct
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_REORDER_FROM_RAW = [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]


def validate_action_chunk(actions, max_abs_arm_q: float = 3.5,
                          max_abs_gripper_q: float = 5.5,
                          max_step: Optional[float] = None,
                          previous_arm_q: Optional[np.ndarray] = None):
    """Validate a policy action chunk before execution."""
    actions = np.asarray(actions, dtype=float)
    if actions.size == 0:
        raise ValueError("empty action chunk")
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"expected action chunk shape (T, 16), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("action chunk contains non-finite values")
    arm_values = np.concatenate([actions[:, 0:7], actions[:, 8:15]], axis=1)
    gripper_values = actions[:, [7, 15]]
    if np.max(np.abs(arm_values)) > max_abs_arm_q:
        raise ValueError(f"action chunk exceeds arm joint limit {max_abs_arm_q}")
    if np.max(np.abs(gripper_values)) > max_abs_gripper_q:
        raise ValueError(
            f"action chunk exceeds gripper joint limit {max_abs_gripper_q}"
        )
    if max_step is not None and previous_arm_q is not None:
        prev = np.asarray(previous_arm_q, dtype=float).reshape(14)
        first_arm = PolicyAdapter.qpos_action_to_g1_action(actions[0])[0]
        if float(np.max(np.abs(first_arm - prev))) > max_step:
            raise ValueError(f"first action step exceeds limit {max_step}")
    return actions


def encode_jpeg_b64(first_frame_np: np.ndarray) -> str:
    img = np.asarray(first_frame_np)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode JPEG for inference")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class _StdlibWebSocket:
    """Minimal RFC6455 client without external websocket dependencies."""

    def __init__(self, url: str, timeout: float):
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(timeout)
        sock.sendall(req.encode("ascii"))
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise RuntimeError(f"WebSocket handshake failed: empty reply from {url}")
            header += chunk
        status = header.split(b"\r\n", 1)[0]
        if b"101" not in status:
            sock.close()
            raise RuntimeError(f"WebSocket handshake failed: {status.decode('ascii', 'replace')}")
        expect = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        )
        if expect not in header:
            logger.warning("WebSocket accept key mismatch from %s", url)
        self._sock = sock

    def settimeout(self, timeout):
        self._sock.settimeout(timeout)

    def send(self, text: str):
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(header + masked)

    def recv(self) -> str:
        def read_exact(n):
            buf = bytearray()
            while len(buf) < n:
                chunk = self._sock.recv(n - len(buf))
                if not chunk:
                    raise RuntimeError("WebSocket closed while reading")
                buf.extend(chunk)
            return bytes(buf)

        b0, b1 = read_exact(2)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack("!H", read_exact(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", read_exact(8))[0]
        mask = read_exact(4) if masked else b""
        payload = bytearray(read_exact(n))
        if masked:
            payload = bytearray(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:
            raise RuntimeError("WebSocket closed by server")
        if opcode == 0x9:
            self._send_pong(bytes(payload)[:125])
            return self.recv()
        if opcode != 0x1:
            raise RuntimeError(f"unsupported WebSocket opcode {opcode}")
        return bytes(payload).decode("utf-8")

    def _send_pong(self, payload: bytes):
        mask = os.urandom(4)
        header = bytearray([0x8A, 0x80 | len(payload)])
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(header + masked)

    def close(self):
        try:
            self._sock.sendall(b"\x88\x80\x00\x00\x00\x00")
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass


class PolicyRemoteClient:
    """Remote policy inference client (WebSocket or ZMQ)."""

    def __init__(self, server_host: str, server_port: int, timeout_ms: int = 60000,
                 auto_connect: bool = True, protocol: str = "ws"):
        self.protocol = str(protocol or "ws").lower()
        if self.protocol == "websocket":
            self.protocol = "ws"
        if self.protocol not in ("ws", "zmq"):
            raise ValueError(f"unsupported protocol {protocol!r}, use ws or zmq")
        self.server_host = server_host
        self.server_port = server_port
        self.url = f"ws://{server_host}:{server_port}/ws"
        self._zmq_endpoint = f"tcp://{server_host}:{server_port}"
        self.timeout_s = max(1.0, timeout_ms / 1000.0)
        self._timeout_ms = int(timeout_ms)
        self._ws = None
        self._zmq_ctx = None
        self._zmq_sock = None
        self._logged_shape = False
        self._closed = False
        if auto_connect:
            self.connect()

    def connect(self):
        if self._closed:
            raise RuntimeError("policy client is closed")
        if self.protocol == "zmq":
            # ZMQ sockets are thread-affine. Each predict call creates and closes
            # its own socket in the inference worker thread.
            logger.info("Policy server configured at %s (zmq)", self._zmq_endpoint)
            return
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        try:
            from websocket import create_connection
            self._ws = create_connection(self.url, timeout=self.timeout_s)
        except ImportError:
            self._ws = _StdlibWebSocket(self.url, timeout=self.timeout_s)
        logger.info("Connected to policy server at %s (ws)", self.url)

    def _close_zmq(self):
        if self._zmq_sock is not None:
            try:
                self._zmq_sock.close()
            except Exception:
                pass
            self._zmq_sock = None
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
            self._zmq_ctx = None

    def close(self):
        self._closed = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._close_zmq()

    def _finalize_actions(self, actions, predict_ms):
        actions = np.asarray(actions, dtype=np.float32)
        predict_ms = float(predict_ms)
        if actions.ndim != 2 or actions.shape[-1] != 16:
            raise RuntimeError(f"expected action chunk (*, 16), got {actions.shape}")
        if not self._logged_shape:
            self._logged_shape = True
            logger.info(
                "first chunk shape=%s dtype=%s predict=%.1fms",
                actions.shape, actions.dtype, predict_ms,
            )
        return actions, predict_ms

    def _predict_zmq(self, first_frame_np, state_np, instruction):
        if self._closed:
            raise RuntimeError("policy client is closed")
        import zmq

        req = {
            "first_frame": np.asarray(first_frame_np),
            "state": np.asarray(state_np, dtype=np.float32),
            "instruction": instruction,
            "rtc_prev": None,
            "rtc_inference_delay": None,
        }
        last_err = None
        for _ in range(2):
            sock = None
            try:
                if self._closed:
                    raise RuntimeError("policy client is closed")
                ctx = zmq.Context.instance()
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
                sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
                sock.connect(self._zmq_endpoint)
                sock.send(pickle.dumps(req))
                resp = pickle.loads(sock.recv())
                break
            except Exception as exc:
                last_err = exc
                if self._closed:
                    raise RuntimeError("policy client closed during inference") from exc
                logger.warning("ZMQ request failed (%s), retrying", exc)
            finally:
                if sock is not None:
                    sock.close()
        else:
            raise RuntimeError(f"ZMQ inference failed: {last_err}") from last_err
        if resp.get("status") != "ok":
            raise RuntimeError(f"ZMQ server error: {resp.get('message', 'unknown')}")
        return self._finalize_actions(resp["actions"], resp.get("predict_ms", 0.0))

    def predict(self, first_frame_np, state_np, instruction):
        if self.protocol == "zmq":
            return self._predict_zmq(first_frame_np, state_np, instruction)
        payload = {
            "type": "inference",
            "instruction": instruction,
            "image": encode_jpeg_b64(first_frame_np),
            "state": np.asarray(state_np, dtype=np.float32).reshape(-1).tolist(),
            "auto_find_t5_embeddings": True,
        }
        if self._closed:
            raise RuntimeError("policy client is closed")
        raw = json.dumps(payload)
        last_err = None
        for _ in range(2):
            try:
                if self._ws is None:
                    self.connect()
                self._ws.settimeout(self.timeout_s)
                self._ws.send(raw)
                resp = self._ws.recv()
                if not resp:
                    raise RuntimeError("empty WebSocket reply")
                data = json.loads(resp)
                break
            except Exception as exc:
                last_err = exc
                if self._closed:
                    raise RuntimeError("policy client closed during inference") from exc
                logger.warning("WebSocket send failed (%s), reconnecting", exc)
                self._ws = None
        else:
            raise RuntimeError(f"WebSocket inference failed: {last_err}") from last_err

        if data.get("type") == "error":
            raise RuntimeError(f"WebSocket server error: {data.get('detail')}")
        if data.get("type") not in (None, "inference"):
            raise RuntimeError(f"unexpected WebSocket reply: {data.get('type')}")
        return self._finalize_actions(
            data["predicted_actions"], data.get("processing_time_ms", 0.0)
        )


@dataclass
class PolicyExecStep:
    arm_q: np.ndarray
    left_grip: float
    right_grip: float


class PolicyAdapter:
    """Convert camera/state observations to policy requests and rows to robot commands."""

    def __init__(self, config_path: str, instruction: str = "",
                 swap_wrists: Optional[bool] = None,
                 pad_joint_values: Optional[str] = None,
                 action_interp_factor: int = 2,
                 exec_chunk_steps: int = 8):
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config_dict = yaml.safe_load(handle)

        common = self.config_dict["common"]
        ds_cfg = self.config_dict.get("dataset", {}) or {}
        self._video_height = int(common["video_height"])
        self._video_width = int(common["video_width"])
        self._stitch_mode = str(ds_cfg.get("stitch_mode", "aspect"))
        self.action_mode = str(ds_cfg.get("action_mode", "qpos"))
        self.current_instruction = instruction
        self.action_interp_factor = max(1, int(action_interp_factor))
        self.exec_chunk_steps = max(0, int(exec_chunk_steps))
        self.swap_wrists = (
            os.environ.get("SWAP_WRISTS", "1") != "0"
            if swap_wrists is None else bool(swap_wrists)
        )

        pad_dims = [int(d) for d in (common.get("action_padding_dims") or [])]
        model_to_arm = {i: i for i in range(7)}
        model_to_arm.update({8 + i: 7 + i for i in range(7)})
        self.hold_arm_joints = sorted(
            model_to_arm[d] for d in pad_dims if d in model_to_arm
        )
        self.pad_joint_values = {}
        default_pad = "6:0.0831,13:-0.1447" if pad_dims else ""
        pad_env = pad_joint_values if pad_joint_values is not None else os.environ.get(
            "PAD_JOINT_VALUES", default_pad
        )
        for item in pad_env.split(","):
            item = item.strip()
            if not item:
                continue
            idx, _, val = item.partition(":")
            self.pad_joint_values[int(idx)] = float(val)

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction

    @staticmethod
    def _to_arm_interleaved(vec: np.ndarray) -> np.ndarray:
        return vec[_REORDER_FROM_RAW]

    @staticmethod
    def _from_arm_interleaved(vec: np.ndarray) -> np.ndarray:
        out = np.empty(16, dtype=vec.dtype)
        out[0:7] = vec[0:7]
        out[14] = vec[7]
        out[7:14] = vec[8:15]
        out[15] = vec[15]
        return out

    @staticmethod
    def qpos_action_to_g1_action(qpos_action_16):
        raw = PolicyAdapter._from_arm_interleaved(np.asarray(qpos_action_16, dtype=float))
        arm_action = np.concatenate([raw[0:7], raw[7:14]])
        return arm_action, float(raw[14]), float(raw[15])

    def _resize_with_padding(self, img, target_hw):
        th, tw = target_hw
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
        y0 = (th - new_h) // 2
        x0 = (tw - new_w) // 2
        if img.ndim == 3:
            canvas[y0:y0 + new_h, x0:x0 + new_w, :] = resized
        else:
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _t_shape_geometry(self, top_hw, bl_hw, br_hw):
        th, tw = self._video_height, self._video_width
        split_w = tw // 2
        right_w = tw - split_w

        def _natural_h(hw, target_w):
            h, w = hw
            return max(1, int(round(h * target_w / w)))

        bot_h = max(_natural_h(bl_hw, split_w), _natural_h(br_hw, right_w))
        bot_h = min(bot_h, th - 1)
        top_h = min(_natural_h(top_hw, tw), th - bot_h)
        natural_h = top_h + bot_h
        pad_top = (th - natural_h) // 2
        return {
            "top_h": top_h, "bot_h": bot_h,
            "split_w": split_w, "right_w": right_w,
            "pad_top": pad_top, "pad_bottom": th - natural_h - pad_top,
        }

    def build_stitched_image(self, head_img, left_wrist_img, right_wrist_img):
        th, tw = self._video_height, self._video_width
        if self._stitch_mode != "aspect":
            raise NotImplementedError("only aspect stitch_mode is supported")
        geo = self._t_shape_geometry(
            head_img.shape[:2], left_wrist_img.shape[:2], right_wrist_img.shape[:2]
        )
        top_h, bot_h = geo["top_h"], geo["bot_h"]
        split_w, right_w = geo["split_w"], geo["right_w"]
        y0 = geo["pad_top"]

        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas[y0:y0 + top_h, :, :] = self._resize_with_padding(head_img, (top_h, tw))
        yb = y0 + top_h
        canvas[yb:yb + bot_h, :split_w, :] = self._resize_with_padding(
            left_wrist_img, (bot_h, split_w)
        )
        canvas[yb:yb + bot_h, split_w:, :] = self._resize_with_padding(
            right_wrist_img, (bot_h, right_w)
        )
        return canvas.astype(np.float32) / 255.0

    def build_model_input(self, head_bgr, left_wrist_bgr, right_wrist_bgr,
                          current_arm_q, left_grip, right_grip):
        if head_bgr is None or left_wrist_bgr is None or right_wrist_bgr is None:
            raise ValueError("missing required camera frame")
        head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
        left_rgb = cv2.cvtColor(left_wrist_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_wrist_bgr, cv2.COLOR_BGR2RGB)
        if self.swap_wrists:
            left_rgb, right_rgb = right_rgb, left_rgb
        stitched = self.build_stitched_image(head_rgb, left_rgb, right_rgb)
        first_frame_np = (stitched * 255).astype(np.uint8)

        if self.action_mode != "qpos":
            raise NotImplementedError("policy handoff supports qpos action mode only")
        raw_state = np.zeros(16, dtype=np.float32)
        raw_state[0:7] = current_arm_q[0:7]
        raw_state[7:14] = current_arm_q[7:14]
        raw_state[14] = float(left_grip)
        raw_state[15] = float(right_grip)
        state = self._to_arm_interleaved(raw_state)
        return first_frame_np, state.astype(np.float32)

    def interpolate_chunk(self, actions):
        actions = np.asarray(actions, dtype=float)
        if self.action_interp_factor <= 1:
            return actions
        steps, dims = actions.shape
        source = np.arange(steps, dtype=np.float64)
        target = np.linspace(0.0, steps - 1, steps * self.action_interp_factor)
        interpolated = np.empty((target.size, dims), dtype=actions.dtype)
        for dim in range(dims):
            interpolated[:, dim] = np.interp(target, source, actions[:, dim])
        return interpolated

    def build_exec_queue(self, raw_actions, current_arm_q):
        raw_actions = validate_action_chunk(raw_actions, previous_arm_q=current_arm_q)
        if 0 < self.exec_chunk_steps < raw_actions.shape[0]:
            raw_actions = raw_actions[:self.exec_chunk_steps]
        exec_actions = self.interpolate_chunk(raw_actions)
        queue = []
        for row in exec_actions:
            arm_q, left_grip, right_grip = self.qpos_action_to_g1_action(row)
            for j in self.hold_arm_joints:
                arm_q[j] = self.pad_joint_values.get(j, current_arm_q[j])
            queue.append(PolicyExecStep(
                arm_q=np.asarray(arm_q, dtype=float).copy(),
                left_grip=left_grip,
                right_grip=right_grip,
            ))
        return queue
