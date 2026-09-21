from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane, Sphere, Line
from multiprocessing import Value, Array, Process, shared_memory
import numpy as np
import asyncio
import threading
import cv2
import os
import time
from pathlib import Path
from typing import Literal


class TeleVuer:
    def __init__(self, use_hand_tracking: bool, binocular: bool=True, img_shape: tuple=None, display_fps: float=30.0,
                       display_mode: Literal["immersive", "pass-through", "ego"]="immersive", zmq: bool=False, webrtc: bool=False, webrtc_url: str=None, 
                       cert_file: str=None, key_file: str=None, alignment_targets=None, alignment_current=None, alignment_state_shared=None):
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages image and pose data.

        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param img_shape: tuple, shape of the head image (height, width).
        :param display_fps: float, target frames per second for display updates (default: 30.0).
        
        :param display_mode: str, controls the VR viewing mode. Options are "immersive", "pass-through", and "ego".
        :param zmq: bool, whether to use zmq for image transmission.
        :param webrtc: bool, whether to use webrtc for real-time communication.
        :param webrtc_url: str, URL for the webrtc offer. must be provided if webrtc is True.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.

        Note:

        - display_mode controls what the VR headset displays:
            * "immersive": fully immersive mode; VR shows the robot's first-person view (zmq or webrtc must be enabled).
            * "pass-through": VR shows the real world through the VR headset cameras; no image from zmq or webrtc is displayed (even if enabled).
            * "ego": a small window in the center shows the robot's first-person view, while the surrounding area shows the real world.
        
        - Only one image mode is active at a time.
        - Image transmission to VR occurs only if display_mode is "immersive" or "ego" and the corresponding zmq or webrtc option is enabled.
        - If zmq and webrtc simultaneously enabled, webrtc will be prioritized.

        --------------              -------------------           --------------       -----------------                     -------
         display_mode       |        display behavior         |    image to VR     |      image source        |               Notes
        --------------              -------------------           --------------       -----------------                     ------- 
           immersive        |   fully immersive view (robot)  |     Yes (full)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------
         pass-through       |       Real world view (VR)      |         No         |          N/A             |  even if image source enabled, don't display
        --------------              -------------------           --------------       -----------------                     -------
              ego           |      ego view (robot + VR)      |    Yes (small)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------

        """
        self.use_hand_tracking = use_hand_tracking
        self.binocular = binocular
        if img_shape is None:
            raise ValueError("[TeleVuer] img_shape must be provided.")
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.display_fps = display_fps
        self.img_height = self.img_shape[0]
        if self.binocular:
            self.img_width  = self.img_shape[1] // 2
        else:
            self.img_width  = self.img_shape[1]
        self.aspect_ratio = self.img_width / self.img_height

        # SSL certificate path resolution
        env_cert = os.getenv("XR_TELEOP_CERT")
        env_key = os.getenv("XR_TELEOP_KEY")
        if cert_file is None or key_file is None:
            if env_cert and env_key:
                cert_file = cert_file or env_cert
                key_file = key_file or env_key
            else:
                unitree_home = Path("/home/unitree")
                user_conf_dir = unitree_home / ".config" / "xr_teleoperate"
                
                cert_path_user = user_conf_dir / "cert.pem"
                key_path_user = user_conf_dir / "key.pem"

                if cert_path_user.exists() and key_path_user.exists():
                    cert_file = cert_file or str(cert_path_user)
                    key_file = key_file or str(key_path_user)
                else:
                    current_module_dir = Path(__file__).resolve().parent.parent.parent
                    cert_file = cert_file or str(current_module_dir / "cert.pem")
                    key_file = key_file or str(current_module_dir / "key.pem")

        self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=10)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        self.display_mode = display_mode
        self.zmq = zmq
        self.webrtc = webrtc
        self.webrtc_url = webrtc_url

        if self.display_mode == "immersive":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc if self.binocular else self.main_image_monocular_webrtc
            elif self.zmq:
                self.img2display_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                fn = self.main_image_binocular_zmq if self.binocular else self.main_image_monocular_zmq
            else:
                raise ValueError("[TeleVuer] immersive mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "ego":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc_ego if self.binocular else self.main_image_monocular_webrtc_ego
            elif self.zmq:
                self.img2display_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                fn = self.main_image_binocular_zmq_ego if self.binocular else self.main_image_monocular_zmq_ego
            else:
                raise ValueError("[TeleVuer] ego mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "pass-through":
            fn = self.main_pass_through
        else:
            raise ValueError(f"[TeleVuer] Unknown display_mode: {self.display_mode}")
        
        self.alignment_targets_shared = None
        self.alignment_current_shared = None
        self.alignment_state_shared = alignment_state_shared
        self.alignment_enabled = alignment_targets is not None
        self._display_warning_times = {}
        self._alignment_marker_logged = False
        if self.alignment_enabled:
            self.alignment_targets_shared = Array('d', 32, lock=True)
            with self.alignment_targets_shared.get_lock():
                self.alignment_targets_shared[:] = np.asarray(alignment_targets, dtype=float).reshape(-1, order="F")
            self.alignment_current_shared = Array('d', 32, lock=True)
            initial_current = alignment_current if alignment_current is not None else alignment_targets
            with self.alignment_current_shared.get_lock():
                self.alignment_current_shared[:] = np.asarray(initial_current, dtype=float).reshape(-1, order="F")
        self.vuer.spawn(start=False)(fn)

        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        self.motion_data_ready_shared = Value('b', False, lock=True)
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_hand_pinch_shared = Value('b', False, lock=True)
            self.left_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.left_hand_squeeze_shared = Value('b', False, lock=True)
            self.left_hand_squeezeValue_shared = Value('d', 0.0, lock=True)

            self.right_hand_pinch_shared = Value('b', False, lock=True)
            self.right_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.right_hand_squeeze_shared = Value('b', False, lock=True)
            self.right_hand_squeezeValue_shared = Value('d', 0.0, lock=True)
        else:
            self.left_ctrl_trigger_shared = Value('b', False, lock=True)
            self.left_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.left_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.left_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.left_ctrl_aButton_shared = Value('b', False, lock=True)
            self.left_ctrl_bButton_shared = Value('b', False, lock=True)

            self.right_ctrl_trigger_shared = Value('b', False, lock=True)
            self.right_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.right_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.right_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.right_ctrl_aButton_shared = Value('b', False, lock=True)
            self.right_ctrl_bButton_shared = Value('b', False, lock=True)

        self.process = Process(target=self._vuer_run)
        self.process.daemon = True
        self.process.start()
    
    def _vuer_run(self):
        try:
            self.vuer.run()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Vuer encountered an error: {e}")
        finally:
            if hasattr(self, "stop_writer_event"):
                self.stop_writer_event.set()

    def _xr_render_loop(self):
        while not self.stop_writer_event.is_set():
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()
            if self.latest_frame is None:
                continue
            latest_frame = self.latest_frame
            latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            self.img2display[:] = latest_frame
    
    def set_alignment_targets(self, left, right):
        """Atomically update dynamic XR-world alignment marker targets."""
        if self.alignment_targets_shared is None:
            return
        targets = np.stack([np.asarray(left, dtype=float).reshape(4, 4),
                            np.asarray(right, dtype=float).reshape(4, 4)])
        if not np.all(np.isfinite(targets)):
            raise ValueError("alignment targets must contain finite values")
        if not np.allclose(targets[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("alignment targets must be homogeneous transforms")
        with self.alignment_targets_shared.get_lock():
            self.alignment_targets_shared[:] = targets.reshape(-1, order="F")

    def set_alignment_current_targets(self, left, right):
        """Atomically update live controller/EEF frames shown in XR."""
        if self.alignment_current_shared is None:
            return
        current = np.stack([np.asarray(left, dtype=float).reshape(4, 4),
                            np.asarray(right, dtype=float).reshape(4, 4)])
        if not np.all(np.isfinite(current)):
            raise ValueError("current alignment poses must contain finite values")
        if not np.allclose(current[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("current alignment poses must be homogeneous transforms")
        with self.alignment_current_shared.get_lock():
            self.alignment_current_shared[:] = current.reshape(-1, order="F")

    def render_to_xr(self, image):
        if self.webrtc or self.display_mode == "pass-through":
            print("[TeleVuer] Warning: render_to_xr is ignored when webrtc is enabled or pass_through is True.")
            return
        self.latest_frame = image
        self.new_frame_event.set()

    def close(self):
        self.process.terminate()
        self.process.join(timeout=0.5)
        if self.display_mode in ("immersive", "ego") and not self.webrtc:
            self.stop_writer_event.set()
            self.new_frame_event.set()
            self.writer_thread.join(timeout=0.5)
            try:
                self.img2display_shm.close()
                self.img2display_shm.unlink()
            except:
                pass

    async def on_cam_move(self, event, session, fps=60):
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        """https://docs.vuer.ai/en/latest/examples/20_motion_controllers.html"""
        try:
            # ControllerData
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]
            # ControllerState
            left_controller = event.value["leftState"]
            right_controller = event.value["rightState"]

            def extract_controllers(controllerState, prefix):
                # trigger
                with getattr(self, f"{prefix}_ctrl_trigger_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_trigger_shared").value = bool(controllerState.get("trigger", False))
                with getattr(self, f"{prefix}_ctrl_triggerValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_triggerValue_shared").value = float(controllerState.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_ctrl_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeeze_shared").value = bool(controllerState.get("squeeze", False))
                with getattr(self, f"{prefix}_ctrl_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeezeValue_shared").value = float(controllerState.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"{prefix}_ctrl_thumbstick_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstick_shared").value = bool(controllerState.get("thumbstick", False))
                with getattr(self, f"{prefix}_ctrl_thumbstickValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstickValue_shared")[:] = controllerState.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"{prefix}_ctrl_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_aButton_shared").value = bool(controllerState.get("aButton", False))
                with getattr(self, f"{prefix}_ctrl_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_bButton_shared").value = bool(controllerState.get("bButton", False))

            extract_controllers(left_controller, "left")
            extract_controllers(right_controller, "right")
            with self.motion_data_ready_shared.get_lock():
                self.motion_data_ready_shared.value = True
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        """https://docs.vuer.ai/en/latest/examples/19_hand_tracking.html"""
        try:
            # HandsData
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand = event.value["leftState"]
            right_hand = event.value["rightState"]
            # HandState
            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]

            def extract_hands(handState, prefix):
                # pinch
                with getattr(self, f"{prefix}_hand_pinch_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinch_shared").value = bool(handState.get("pinch", False))
                with getattr(self, f"{prefix}_hand_pinchValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_pinchValue_shared").value = float(handState.get("pinchValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_hand_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeeze_shared").value = bool(handState.get("squeeze", False))
                with getattr(self, f"{prefix}_hand_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_hand_squeezeValue_shared").value = float(handState.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self.left_arm_pose_shared, self.left_hand_position_shared, self.left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self.right_arm_pose_shared, self.right_hand_position_shared, self.right_hand_orientation_shared)
            extract_hands(left_hand, "left")
            extract_hands(right_hand, "right")
            with self.motion_data_ready_shared.get_lock():
                self.motion_data_ready_shared.value = True

        except Exception:
            pass
    
    def _warn_display_update(self, label, error, session_missing=False):
        now = time.monotonic()
        last = self._display_warning_times.get(label, 0.0)
        if now - last >= 5.0:
            self._display_warning_times[label] = now
            action = "ending handler" if session_missing else "retrying"
            print(f"[TeleVuer] Warning: {label} upsert failed; {action}: {error}")

    @staticmethod
    def _is_session_missing_error(error):
        error_text = str(error)
        return "Websocket session is missing" in error_text or "CURRENT_WS_ID" in error_text

    def _safe_upsert(self, session, *args, **kwargs):
        try:
            session.upsert(*args, **kwargs)
            return True
        except Exception as error:
            self._warn_display_update(
                "display", error, session_missing=self._is_session_missing_error(error)
            )
            return False

    async def _upsert_alignment_markers(self, session):
        if not self.alignment_enabled or self.alignment_targets_shared is None:
            return True
        try:
            with self.alignment_targets_shared.get_lock():
                targets = np.array(self.alignment_targets_shared[:], dtype=float).reshape(2, 4, 4, order="F")
            with self.alignment_current_shared.get_lock():
                current = np.array(self.alignment_current_shared[:], dtype=float).reshape(2, 4, 4, order="F")
            state = 0
            if self.alignment_state_shared is not None:
                with self.alignment_state_shared.get_lock():
                    state = int(self.alignment_state_shared.value if hasattr(self.alignment_state_shared, "value") else self.alignment_state_shared[0])
            # 0 ALIGNING, 1 ALIGNED, 2 ACTIVE. State is shared explicitly across processes.
            color = ("#ff8800", "#33dd55", "#3388ff")[max(0, min(state, 2))]
            objects = []
            axis_sizes = ([0.16, 0.012, 0.012], [0.012, 0.16, 0.012], [0.012, 0.012, 0.16])
            axis_colors = ("#ff2020", "#20dd40", "#2080ff")  # X/Y/Z = red/green/blue
            for side, target, current_pose in (("left", targets[0], current[0]), ("right", targets[1], current[1])):
                # In LIVE/ACTIVE, show only the current EEF frame. During
                # ALIGNING, show the frozen TARGET and the live controller frame.
                show_target = state != 2
                if show_target:
                    target_matrix = target.reshape(-1, order="F").tolist()
                    objects.append(Sphere(matrix=target_matrix, position=target[:3, 3].tolist(),
                                          args=[0.055, 16, 16], material={"color": color},
                                          key=f"alignment-{side}-target-center"))
                    for axis in range(3):
                        axis_end = target[:3, 3] + target[:3, axis] * 0.16
                        objects.append(Line(
                            points=[target[:3, 3].tolist(), axis_end.tolist()],
                            color=axis_colors[axis], lineWidth=8,
                            key=f"alignment-{side}-target-axis-{axis}"
                        ))
                current_matrix = current_pose.reshape(-1, order="F").tolist()
                objects.append(Sphere(matrix=current_matrix, position=current_pose[:3, 3].tolist(),
                                      args=[0.04 if show_target else 0.055, 12, 12],
                                      material={"color": "#00e5ff" if show_target else color},
                                      key=f"alignment-{side}-current-center"))
                for axis in range(3):
                    axis_end = current_pose[:3, 3] + current_pose[:3, axis] * (0.12 if show_target else 0.16)
                    objects.append(Line(
                        points=[current_pose[:3, 3].tolist(), axis_end.tolist()],
                        color=axis_colors[axis], lineWidth=8,
                        key=f"alignment-{side}-current-axis-{axis}"
                    ))
            session.upsert(objects, to="bgChildren")
            if not self._alignment_marker_logged:
                self._alignment_marker_logged = True
                print("[TeleVuer] Alignment markers upsert succeeded.")
            return True
        except Exception as error:
            self._warn_display_update(
                "alignment marker", error, session_missing=self._is_session_missing_error(error)
            )
            return False

    ## immersive MODE
    async def main_image_binocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        while True:
            if not self._safe_upsert(session,
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width],
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=80,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if not self._safe_upsert(session,
                [
                    ImageBackground(
                        self.img2display,
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=80,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if not self._safe_upsert(session,
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 7,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if not self._safe_upsert(session,
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 7,
                ),
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            await asyncio.sleep(1.0 / self.display_fps)

    ## ego MODE
    async def main_image_binocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        while True:
            if not self._safe_upsert(session,
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=80,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        layers=2,
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if not self._safe_upsert(session,
                [
                    ImageBackground(
                        self.img2display,
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        format="jpeg",
                        quality=80,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if not self._safe_upsert(session,
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    fps=90,
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            if not self._safe_upsert(session,
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                ),
                to="bgChildren",
            ):
                return
            if not await self._upsert_alignment_markers(session):
                return
            await asyncio.sleep(1.0 / self.display_fps)

    ## pass-through MODE
    async def main_pass_through(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    fps=90,
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        # In controller mode, do not add the Vuer controller meshes to the
        # pass-through scene; pose data is still received by the event handler.

        while True:
            if not await self._upsert_alignment_markers(session):
                return
            await asyncio.sleep(1.0 / self.display_fps)

    # ==================== common data ====================
    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch(self):
        """bool, whether left hand is pinching."""
        with self.left_hand_pinch_shared.get_lock():
            return self.left_hand_pinch_shared.value

    @property
    def left_hand_pinchValue(self):
        """float, pinch strength of left hand."""
        with self.left_hand_pinchValue_shared.get_lock():
            return self.left_hand_pinchValue_shared.value

    @property
    def left_hand_squeeze(self):
        """bool, whether left hand is squeezing."""
        with self.left_hand_squeeze_shared.get_lock():
            return self.left_hand_squeeze_shared.value

    @property
    def left_hand_squeezeValue(self):
        """float, squeeze strength of left hand."""
        with self.left_hand_squeezeValue_shared.get_lock():
            return self.left_hand_squeezeValue_shared.value

    @property
    def right_hand_pinch(self):
        """bool, whether right hand is pinching."""
        with self.right_hand_pinch_shared.get_lock():
            return self.right_hand_pinch_shared.value

    @property
    def right_hand_pinchValue(self):
        """float, pinch strength of right hand."""
        with self.right_hand_pinchValue_shared.get_lock():
            return self.right_hand_pinchValue_shared.value

    @property
    def right_hand_squeeze(self):
        """bool, whether right hand is squeezing."""
        with self.right_hand_squeeze_shared.get_lock():
            return self.right_hand_squeeze_shared.value

    @property
    def right_hand_squeezeValue(self):
        """float, squeeze strength of right hand."""
        with self.right_hand_squeezeValue_shared.get_lock():
            return self.right_hand_squeezeValue_shared.value

    # ==================== Controller Data ====================
    @property
    def left_ctrl_trigger(self):
        """bool, left controller trigger pressed or not."""
        with self.left_ctrl_trigger_shared.get_lock():
            return self.left_ctrl_trigger_shared.value

    @property
    def left_ctrl_triggerValue(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_triggerValue_shared.get_lock():
            return self.left_ctrl_triggerValue_shared.value

    @property
    def left_ctrl_squeeze(self):
        """bool, left controller squeeze pressed or not."""
        with self.left_ctrl_squeeze_shared.get_lock():
            return self.left_ctrl_squeeze_shared.value

    @property
    def left_ctrl_squeezeValue(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_squeezeValue_shared.get_lock():
            return self.left_ctrl_squeezeValue_shared.value

    @property
    def left_ctrl_thumbstick(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self.left_ctrl_thumbstick_shared.get_lock():
            return self.left_ctrl_thumbstick_shared.value

    @property
    def left_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self.left_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.left_ctrl_thumbstickValue_shared[:])

    @property
    def left_ctrl_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self.left_ctrl_aButton_shared.get_lock():
            return self.left_ctrl_aButton_shared.value

    @property
    def left_ctrl_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self.left_ctrl_bButton_shared.get_lock():
            return self.left_ctrl_bButton_shared.value

    @property
    def right_ctrl_trigger(self):
        """bool, right controller trigger pressed or not."""
        with self.right_ctrl_trigger_shared.get_lock():
            return self.right_ctrl_trigger_shared.value

    @property
    def right_ctrl_triggerValue(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_triggerValue_shared.get_lock():
            return self.right_ctrl_triggerValue_shared.value

    @property
    def right_ctrl_squeeze(self):
        """bool, right controller squeeze pressed or not."""
        with self.right_ctrl_squeeze_shared.get_lock():
            return self.right_ctrl_squeeze_shared.value

    @property
    def right_ctrl_squeezeValue(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_squeezeValue_shared.get_lock():
            return self.right_ctrl_squeezeValue_shared.value

    @property
    def right_ctrl_thumbstick(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self.right_ctrl_thumbstick_shared.get_lock():
            return self.right_ctrl_thumbstick_shared.value

    @property
    def right_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self.right_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.right_ctrl_thumbstickValue_shared[:])

    @property
    def right_ctrl_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self.right_ctrl_aButton_shared.get_lock():
            return self.right_ctrl_aButton_shared.value

    @property
    def right_ctrl_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self.right_ctrl_bButton_shared.get_lock():
            return self.right_ctrl_bButton_shared.value

    @property
    def motion_data_ready(self):
        """bool, whether at least one hand or controller motion data event has been received."""
        with self.motion_data_ready_shared.get_lock():
            return self.motion_data_ready_shared.value
