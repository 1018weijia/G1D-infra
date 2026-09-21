from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber # dds
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_ # idl
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point32_,geometry_msgs_msg_dds__Twist_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
import threading
import time
from multiprocessing import Process, Array

import logging_mp
logger_mp = logging_mp.getLogger(__name__)
from teleop.robot_control.dds_utils import wait_for_dds



kTopicHeightCmd = "rt/cmd_hispeed"
kTopicHeightState = "rt/hispeed_state"
kTopicG1MoveCmd = "rt/cmd_vel_no_limit"
kTopicG1MoveState = "rt/slamware_ros_sdk_server_node/odom"

kTopicR3Controller = "rt/wirelesscontroller"

class G1_Mobile_Lift_Controller:
    def __init__(self, base_type, r3_controller, fps = 30.0):
        """
        Initialize G1 mobile base and elevation controller
        
        Args:
            base_type: "only_height" for height only, "with_move" for height + movement
            r3_controller: "true" for R3 controller, "false" for XR controller
            fps: Control frequency
        """
        logger_mp.info("Initialize G1_Mobile_Lift_Controller...")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        self.base_type = base_type 
        self.r3_controller = r3_controller
        # Data reception flags
        self.height_data_received = False
        self.move_data_received = False

        # init buffer
        # For controlling height and movement commands
        self.g1_height_action_array_in = Array('d', 1, lock = True) 
        self.g1_move_action_array_in = Array('d', 2, lock = True)


        # For receiving height and movement state
        self.g1_height_state_array_out  = Array('d', 2, lock=True)  
        self.g1_height_action_array_out = Array('d', 1, lock=True)  # For receiving published height action values, ready to save to dataset
        self.g1_move_state_array_out = None
        self.g1_move_action_array_out = None  # For receiving published movement action values, ready to save to dataset

        self.r3_controller_state_array_out = None

        # Height control publisher
        self.HeightCmb_publisher = ChannelPublisher(kTopicHeightCmd, Point32_)
        self.HeightCmb_publisher.Init()
        # Height state subscriber
        self.HeightState_subscriber = ChannelSubscriber(kTopicHeightState, Point32_)
        self.HeightState_subscriber.Init()

        self.g1_height_msg = geometry_msgs_msg_dds__Point32_()

        # When base_type is with_move, use movement control; otherwise only use height control
        if self.base_type == "mobile_lift":
            self.g1_move_state_array_out = Array('d', 2, lock=True)
            self.g1_move_action_array_out = Array('d', 2, lock=True)  # For receiving published movement action values, ready to save to dataset
            # Movement control publisher
            self.G1MoveCmb_publisher = ChannelPublisher(kTopicG1MoveCmd, Twist_)
            self.G1MoveCmb_publisher.Init()
            self.g1_move_msg = geometry_msgs_msg_dds__Twist_()
            # Movement state subscriber
            self.G1MoveState_subscriber = ChannelSubscriber(kTopicG1MoveState, Odometry_)
            self.G1MoveState_subscriber.Init()

        self.subscribe_g1_mobilebase_state_thread = threading.Thread(target=self._subscribe_g1_mobilebase_state)
        self.subscribe_g1_mobilebase_state_thread.daemon = True
        self.subscribe_g1_mobilebase_state_thread.start()

        wait_for_dds(
            lambda: self.height_data_received and (self.base_type != "mobile_lift" or self.move_data_received),
            "G1_Mobile_Lift_Controller"
        )
        # If not using Unitree controller, start control process
        if self.r3_controller:
            self.r3_controller_state_array_out = Array('d', 5, lock=True)
            self.r3_controller_state_subscriber = ChannelSubscriber(kTopicR3Controller, WirelessController_)
            self.r3_controller_state_subscriber.Init()
            self.subscribe_r3_controller_state_thread = threading.Thread(target=self._subscribe_r3_controller_state)
            self.subscribe_r3_controller_state_thread.daemon = True
            self.subscribe_r3_controller_state_thread.start()
        self.running = True
        mobile_control_process = Process(target=self.control_process, args=(self.base_type,))
        mobile_control_process.daemon = True
        mobile_control_process.start()

        logger_mp.info("Initialize G1_Mobile_Lift_Controller OK!\n")

    def _subscribe_r3_controller_state(self):
        while True:
            try:
                r3_controller_msg = self.r3_controller_state_subscriber.Read()
                if r3_controller_msg is not None:
                    self.r3_controller_state_array_out[1] = r3_controller_msg.lx
                    self.r3_controller_state_array_out[0] = r3_controller_msg.ly
                    self.r3_controller_state_array_out[2] = r3_controller_msg.rx
                    self.r3_controller_state_array_out[3] = r3_controller_msg.ry
                    self.r3_controller_state_array_out[4] = r3_controller_msg.keys
            except Exception as e:
                logger_mp.info(f"[_subscribe_r3_controller_state] Exception: {e}")
                time.sleep(0.1)
            time.sleep(0.01)

    def _subscribe_g1_mobilebase_state(self):
        while True:
            try:
                height_msg = self.HeightState_subscriber.Read()
                if height_msg is not None:
                    self.g1_height_state_array_out[0] = height_msg.y  # in meters
                    self.g1_height_state_array_out[1] = height_msg.z
                    
                    if not self.height_data_received:
                        self.height_data_received = True
                        
                if self.base_type == "mobile_lift":
                    move_msg = self.G1MoveState_subscriber.Read()
                    if move_msg is not None:
                        self.g1_move_state_array_out[0] = move_msg.twist.twist.linear.x
                        self.g1_move_state_array_out[1] = move_msg.twist.twist.angular.z
                        
                        if not self.move_data_received:
                            self.move_data_received = True

                time.sleep(0.01)
                
            except Exception as e:
                logger_mp.info(f"[_subscribe_g1_mobilebase_state] Exception: {e}")
                time.sleep(0.1) 

    def ctrl_g1_height(self, g1_height_target):
        self.g1_height_msg.z = g1_height_target
        self.HeightCmb_publisher.Write(self.g1_height_msg)

    def ctrl_g1_move(self, g1_move_target):
        self.g1_move_msg.linear.x = g1_move_target[0]
        self.g1_move_msg.angular.z = g1_move_target[1]
        self.G1MoveCmb_publisher.Write(self.g1_move_msg)

    def control_process(self, base_type):
        try:
            while self.running:
                self.start_time = time.time()
                target_height = self.g1_height_action_array_in[0]
                self.ctrl_g1_height(target_height)
                
                if base_type == "mobile_lift":
                    g1_move_target = self.g1_move_action_array_in
                    self.ctrl_g1_move(g1_move_target)
                
                current_time = time.time()
                time_elapsed = current_time - self.start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("G1_Mobilebase_Height_Controller has been closed.")
