#!/usr/bin/env python3
"""
HIKROBOT ROS2 - Persistent Connection + Run Once

节点生命周期：
    启动
      -> CreateHandle
      -> OpenDevice
      -> 保持 GigE 控制连接和心跳

机械臂到达 Home：
      /hik_camera/run_once
      -> StartGrabbing
      -> 连续 GetImageBuffer 取 5 帧
      -> 每帧 FreeImageBuffer 并发布
      -> StopGrabbing
      -> 继续保持 OpenDevice

节点退出：
      -> CloseDevice
      -> DestroyHandle

手动测试：
    ros2 topic pub --once /hik_camera/run_once std_msgs/msg/Empty "{}"
"""
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty


def _load_hik_sdk():
    """
    Load HIKROBOT MVS Python wrapper.

    Recommended:
      export HIK_MVS_PYTHON_PATH=/opt/MVS/Samples/64/Python/MvImport

    The exact directory can be found with:
      find /opt/MVS -name MvCameraControl_class.py 2>/dev/null
    """
    candidates = [
        os.environ.get("HIK_MVS_PYTHON_PATH", ""),
        "/opt/MVS/Samples/64/Python/MvImport",
        "/opt/MVS/Samples/64/Python",
    ]
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        # HIKROBOT's official Python samples commonly expose constants,
        # structures and MvCamera through this wrapper.
        module = __import__("MvCameraControl_class", fromlist=["*"]) # pyright: ignore[reportUndefinedVariable]
    except Exception as exc:
        raise RuntimeError(
             "Cannot import HIKROBOT MVS Python SDK. "
            "Run: find /opt/MVS -name MvCameraControl_class.py 2>/dev/null ; "
            "then export HIK_MVS_PYTHON_PATH=<directory-containing-that-file>"
        ) from exc

    return module


HIK = _load_hik_sdk()


class HikCameraNode(Node):
    def __init__(self):
        super().__init__("hik_camera_node")

        self.declare_parameter("device_index", 0)
        self.declare_parameter("publish_fps", 10.0)
        self.declare_parameter("frame_id", "hik_camera_optical_frame")
        self.declare_parameter("image_topic", "/hik_camera/image_raw")

        # ===== Run Once =====
        # 节点启动后相机始终保持 OpenDevice。
        # 机械臂到达 Home 后向 run_once_topic 发布 Empty，
        # 本节点只临时 StartGrabbing -> 获取1帧 -> StopGrabbing。
        self.declare_parameter("run_once_topic", "/hik_camera/run_once")
        self.declare_parameter("run_once_done_topic", "/hik_camera/run_once_done")
        self.declare_parameter("grab_timeout_ms", 3000)

        # 一次 Home / Run Once 任务连续采集并发布的图像帧数。
        # 默认 5 帧，与 svm_detector_node 的 stable_frames=5 对应。
        self.declare_parameter("burst_frames", 5)

        # 节点启动时如果上一条 GigE 控制会话尚未释放，
        # OpenDevice 可能暂时返回 0x80000203。
        # 只在“首次建立长期连接”阶段自动等待重试。
        self.declare_parameter("open_retry_timeout_sec", 15.0)
        self.declare_parameter("open_retry_interval_sec", 0.5)

        self.declare_parameter("force_mono8", False)
        self.declare_parameter("resize_scale", 1.0)
        # ===== Camera parameters =====
        # Gain is enabled now as requested.
        self.declare_parameter("gain_auto", False)
        self.declare_parameter("gain", 23.0)

        # Reserved parameter interfaces for future tuning.
        # These groups are disabled by default and do not change the current
        # camera behaviour unless *_config_enable is set to True.
        self.declare_parameter("exposure_config_enable", False)
        self.declare_parameter("exposure_auto", False)
        self.declare_parameter("exposure_time", 10000.0)

        self.declare_parameter("gamma_config_enable", False)
        self.declare_parameter("gamma_enable", False)
        self.declare_parameter("gamma", 1.0)

        self.declare_parameter("camera_fps_config_enable", False)
        self.declare_parameter("camera_fps", 10.0)

        # ===== GigE heartbeat =====
        # 海康 GigE 相机控制连接的心跳超时时间。
        # 默认设为 10000 ms，用于降低短时系统调度/网络抖动导致的控制连接丢失风险。
        self.declare_parameter("heartbeat_config_enable", True)
        self.declare_parameter("heartbeat_timeout", 3000)

        # False = continuous acquisition; True = trigger mode.
        self.declare_parameter("trigger_mode", False)

        self.device_index = int(self.get_parameter("device_index").value)
        self.publish_fps = float(self.get_parameter("publish_fps").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.image_topic = str(self.get_parameter("image_topic").value)

        self.run_once_topic = str(
            self.get_parameter("run_once_topic").value
        )
        self.run_once_done_topic = str(
            self.get_parameter("run_once_done_topic").value
        )
        self.grab_timeout_ms = int(
            self.get_parameter("grab_timeout_ms").value
        )
        self.burst_frames = int(
            self.get_parameter("burst_frames").value
        )
        self.open_retry_timeout_sec = float(
            self.get_parameter("open_retry_timeout_sec").value
        )
        self.open_retry_interval_sec = float(
            self.get_parameter("open_retry_interval_sec").value
        )

        self.force_mono8 = bool(self.get_parameter("force_mono8").value)
        self.resize_scale = float(self.get_parameter("resize_scale").value)
        self.gain_auto = bool(self.get_parameter("gain_auto").value)
        self.gain = float(self.get_parameter("gain").value)

        self.exposure_config_enable = bool(
            self.get_parameter("exposure_config_enable").value
        )
        self.exposure_auto = bool(self.get_parameter("exposure_auto").value)
        self.exposure_time = float(self.get_parameter("exposure_time").value)

        self.gamma_config_enable = bool(
            self.get_parameter("gamma_config_enable").value
        )
        self.gamma_enable = bool(self.get_parameter("gamma_enable").value)
        self.gamma = float(self.get_parameter("gamma").value)

        self.camera_fps_config_enable = bool(
            self.get_parameter("camera_fps_config_enable").value
        )
        self.camera_fps = float(self.get_parameter("camera_fps").value)

        self.heartbeat_config_enable = bool(
            self.get_parameter("heartbeat_config_enable").value
        )
        self.heartbeat_timeout = int(
            self.get_parameter("heartbeat_timeout").value
        )

        self.trigger_mode = bool(self.get_parameter("trigger_mode").value)


        if self.publish_fps <= 0.0:
            raise ValueError("publish_fps must be > 0")
        if self.resize_scale <= 0.0:
            raise ValueError("resize_scale must be > 0")
        if self.heartbeat_timeout < 0:
            raise ValueError("heartbeat_timeout must be >= 0")
        if self.grab_timeout_ms <= 0:
            raise ValueError("grab_timeout_ms must be > 0")
        if self.burst_frames <= 0:
            raise ValueError("burst_frames must be > 0")
        if self.open_retry_timeout_sec <= 0.0:
            raise ValueError("open_retry_timeout_sec must be > 0")
        if self.open_retry_interval_sec <= 0.0:
            raise ValueError("open_retry_interval_sec must be > 0")

        self.bridge = CvBridge()

        self.pub = self.create_publisher(
            Image,
            self.image_topic,
            max(5, self.burst_frames),
        )

        self.done_pub = self.create_publisher(
            Bool,
            self.run_once_done_topic,
            1,
        )

        self.run_once_sub = self.create_subscription(
            Empty,
            self.run_once_topic,
            self._run_once_callback,
            1,
        )

        self.cam = None
        self.device_opened = False
        self.grabbing = False
        self.capture_busy = False

        # 节点启动时只 OpenDevice，保持 GigE 控制连接。
        # 不在这里 StartGrabbing。
        self._open_camera()

        self.get_logger().info(
            "HIK camera persistent-connection burst node ready. "
            f"Camera remains OPEN; waiting on {self.run_once_topic}. "
            f"burst_frames={self.burst_frames}; "
            f"image={self.image_topic}; done={self.run_once_done_topic}"
        )

    @staticmethod
    def _symbol(name): # 定义获取HIK的name属性的函数
        if not hasattr(HIK, name): # hasattr = has attribute 检查HIK是否有name这个属性
            raise RuntimeError(f"HIK SDK symbol missing: {name}")
        return getattr(HIK, name) # getattr = get attribute 获取HIK的name属性

    def _enumerate_devices(self): # 枚举设备
        MV_CC_DEVICE_INFO_LIST = self._symbol("MV_CC_DEVICE_INFO_LIST")
        MV_GIGE_DEVICE = self._symbol("MV_GIGE_DEVICE")
        MV_USB_DEVICE = self._symbol("MV_USB_DEVICE")
        MvCamera = self._symbol("MvCamera")

        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer = MV_GIGE_DEVICE | MV_USB_DEVICE # transport layer = GigE or USB，通信层
        ret = MvCamera.MV_CC_EnumDevices(tlayer, device_list) # ret = return code，返回值
        if ret != 0:
            raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08x}")

        if device_list.nDeviceNum == 0:
            raise RuntimeError("No HIKROBOT camera found by MVS SDK.")

        self.get_logger().info(f"Found {device_list.nDeviceNum} HIKROBOT camera(s).")
        return device_list

    def _open_camera(self):
        """
        节点启动时建立一次长期相机连接。

        正常流程：
            EnumDevices -> CreateHandle -> OpenDevice
        成功后一直保持 OpenDevice，直到节点退出。

        若 OpenDevice 暂时返回 0x80000203（Access Denied），
        说明上一条 GigE 控制会话可能尚未释放，或相机仍被其他程序占用。
        此时自动重新枚举、重新建句柄并等待重试。
        """
        MvCamera = self._symbol("MvCamera")
        MV_CC_DEVICE_INFO = self._symbol("MV_CC_DEVICE_INFO")
        MV_ACCESS_Exclusive = self._symbol("MV_ACCESS_Exclusive")

        MV_E_ACCESS_DENIED = 0x80000203

        deadline = time.monotonic() + self.open_retry_timeout_sec
        attempt = 0

        while True:
            attempt += 1

            device_list = self._enumerate_devices()

            if (
                self.device_index < 0
                or self.device_index >= device_list.nDeviceNum
            ):
                raise RuntimeError(
                    f"device_index={self.device_index} out of range "
                    f"[0, {device_list.nDeviceNum - 1}]"
                )

            device_info = cast(
                device_list.pDeviceInfo[self.device_index],
                POINTER(MV_CC_DEVICE_INFO),
            ).contents

            cam = MvCamera()

            ret = cam.MV_CC_CreateHandle(device_info)
            if ret != 0:
                raise RuntimeError(
                    f"MV_CC_CreateHandle failed: 0x{ret:08x}"
                )

            ret = cam.MV_CC_OpenDevice(
                MV_ACCESS_Exclusive,
                0,
            )
            ret_u32 = int(ret) & 0xFFFFFFFF

            if ret == 0:
                self.cam = cam
                self.device_opened = True
                self.grabbing = False

                self.get_logger().info(
                    f"Camera device opened successfully "
                    f"(attempt {attempt}) and will remain OPEN."
                )
                break

            # OpenDevice 没成功，这个临时句柄直接销毁。
            try:
                cam.MV_CC_DestroyHandle()
            except Exception:
                pass

            self.cam = None
            self.device_opened = False
            self.grabbing = False

            if ret_u32 != MV_E_ACCESS_DENIED:
                raise RuntimeError(
                    f"MV_CC_OpenDevice failed: 0x{ret_u32:08x}"
                )

            remaining = deadline - time.monotonic()

            if remaining <= 0.0:
                raise RuntimeError(
                    "MV_CC_OpenDevice failed: 0x80000203 "
                    "(ACCESS_DENIED). "
                    f"Waited {self.open_retry_timeout_sec:.1f}s. "
                    "Camera is still occupied. "
                    "Please close MVS or any other camera node/process."
                )

            self.get_logger().warning(
                "MV_CC_OpenDevice = 0x80000203 (ACCESS_DENIED). "
                f"Waiting for previous camera ownership to release: "
                f"attempt={attempt}, remaining={remaining:.1f}s"
            )

            time.sleep(
                min(
                    self.open_retry_interval_sec,
                    max(0.05, remaining),
                )
            )

        # ----------------------------------------------------
        # GigE heartbeat
        # ----------------------------------------------------
        # 先读取相机当前值与合法范围，再根据 ROS 参数决定是否设置。
        heartbeat_info = self._get_int_node("GevHeartbeatTimeout")

        if self.heartbeat_config_enable:
            heartbeat_value = int(self.heartbeat_timeout)

            # 如果成功读取了范围，则将目标值限制到相机合法范围。
            if heartbeat_info is not None:
                minimum = int(heartbeat_info["min"])
                maximum = int(heartbeat_info["max"])
                increment = int(heartbeat_info["inc"])

                if maximum > 0:
                    heartbeat_value = min(heartbeat_value, maximum)
                heartbeat_value = max(heartbeat_value, minimum)

                if increment > 1:
                    heartbeat_value = (
                        minimum
                        + round((heartbeat_value - minimum) / increment) * increment
                    )
                    if maximum > 0:
                        heartbeat_value = min(heartbeat_value, maximum)
                    heartbeat_value = max(heartbeat_value, minimum)

            if self._set_int_node(
                "GevHeartbeatTimeout",
                heartbeat_value,
            ):
                # 再读一次，确认相机实际接受的值。
                self._get_int_node("GevHeartbeatTimeout")
        else:
            self.get_logger().info(
                "Heartbeat configuration disabled; keep camera current value."
            )

        # Configure camera parameters before starting acquisition.
        self._configure_camera_parameters()

        # GigE optimization, when supported.
        try:
            MV_GIGE_DEVICE = self._symbol("MV_GIGE_DEVICE")
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                # packet_size = self.cam.MV_CC_GetOptimalPacketSize()
                packet_size = 1500
                ret = self.cam.MV_CC_SetIntValue(
                    "GevSCPSPacketSize",
                    packet_size
                )

                if ret == 0:
                    self.get_logger().info(
                        f"GigE packet size forced to {packet_size}"
                    )
                else:
                    self.get_logger().warning(
                        f"Failed to set GevSCPSPacketSize={packet_size}, "
                        f"ret=0x{ret:08x}"
                    )
                # if packet_size > 0:
                #     self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                #     self.get_logger().info(
                #         f"GigE optimal packet size set to {packet_size}"
                #     )
        except Exception as exc:
            self.get_logger().warning(f"Could not set GigE packet size: {exc}")

        # Trigger mode. Continuous acquisition remains the default.
        trigger_value = 1 if self.trigger_mode else 0
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", trigger_value)
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set TriggerMode={self.trigger_mode}, ret=0x{ret:08x}"
            )
        else:
            self.get_logger().info(
                f"TriggerMode = {'On' if self.trigger_mode else 'Off'}"
            )

        # Prefer Mono8 for the first ROS integration:
        # lower bandwidth and enough for HOG/Hu/geometry features.
        if self.force_mono8:
            try:
                mono8 = self._symbol("PixelType_Gvsp_Mono8")
                ret = self.cam.MV_CC_SetEnumValue("PixelFormat", mono8)
                if ret != 0:
                    self.get_logger().warning(
                        "Camera rejected PixelFormat=Mono8. "
                        "The node will try SDK pixel conversion."
                    )
            except Exception as exc:
                self.get_logger().warning(f"Could not request Mono8: {exc}")

        # 注意：
        # 节点启动时只保持 OpenDevice，不开始图像流。
        # StartGrabbing 由 /hik_camera/run_once 触发时临时执行。
        self.get_logger().info(
            "Camera configuration completed. "
            "StartGrabbing is idle until a Run Once signal arrives."
        )

    def _get_int_node(self, name):
        """读取 MVS Integer 节点，并打印 current/min/max/inc。"""
        ValueType = getattr(HIK, "MVCC_INTVALUE_EX", None)
        if ValueType is None:
            ValueType = getattr(HIK, "MVCC_INTVALUE", None)

        if ValueType is None:
            self.get_logger().warning(
                f"SDK does not expose MVCC_INTVALUE_EX/MVCC_INTVALUE; "
                f"cannot read {name}."
            )
            return None

        value = ValueType()
        memset(byref(value), 0, sizeof(value))

        ret = self.cam.MV_CC_GetIntValue(name, value)
        if ret != 0:
            self.get_logger().warning(
                f"Failed to read {name}, ret=0x{ret:08x}"
            )
            return None

        current = int(value.nCurValue)
        minimum = int(getattr(value, "nMin", 0))
        maximum = int(getattr(value, "nMax", 0))
        increment = int(getattr(value, "nInc", 0))

        self.get_logger().info(
            f"{name}: current={current}, min={minimum}, "
            f"max={maximum}, inc={increment}"
        )

        return {
            "current": current,
            "min": minimum,
            "max": maximum,
            "inc": increment,
        }

    def _set_int_node(self, name, value):
        """安全设置 MVS Integer 节点。"""
        ret = self.cam.MV_CC_SetIntValue(name, int(value))
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set {name}={int(value)}, ret=0x{ret:08x}"
            )
            return False

        self.get_logger().info(
            f"{name} = {int(value)}"
        )
        return True

    def _set_enum(self, name, value, label=None): # 定义配置摄像头enum型参数的函数
        """Safely set an MVS enum node."""
        ret = self.cam.MV_CC_SetEnumValue(name, int(value)) # 给定的枚举节点设置值
        shown = label if label is not None else str(value)
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set {name}={shown}, ret=0x{ret:08x}"
            )
            return False

        self.get_logger().info(f"{name} = {shown}")
        return True

    def _set_float(self, name, value): # 定义配置摄像头float型参数的函数
        """Safely set an MVS floating-point node."""
        ret = self.cam.MV_CC_SetFloatValue(name, float(value)) # 给定的浮点节点设置值
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set {name}={value}, ret=0x{ret:08x}"
            )
            return False

        self.get_logger().info(f"{name} = {float(value):.3f}")
        return True

    def _set_bool(self, name, value): # 定义配置摄像头bool型参数的函数
        """Safely set an MVS boolean node when supported by the SDK."""
        if not hasattr(self.cam, "MV_CC_SetBoolValue"):
            self.get_logger().warning(
                f"SDK has no MV_CC_SetBoolValue; skip {name}."
            )
            return False

        ret = self.cam.MV_CC_SetBoolValue(name, bool(value)) # 给定的布尔节点设置值
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set {name}={bool(value)}, ret=0x{ret:08x}"
            )
            return False

        self.get_logger().info(f"{name} = {bool(value)}")
        return True

    def _configure_camera_parameters(self): # 配置摄像头参数
        """
        Configure camera parameters before grabbing.

        Active now:
          - GainAuto = Off by default
          - Gain = 23.0 by default

        Reserved interfaces:
          - ExposureAuto / ExposureTime
          - GammaEnable / Gamma
          - AcquisitionFrameRateEnable / AcquisitionFrameRate

        Reserved interfaces are disabled by default.
        """

        # ---------------- Gain ----------------
        # Common MVS/GenICam enum:
        #   0 = Off, 1 = Once, 2 = Continuous
        gain_auto_value = 2 if self.gain_auto else 0
        gain_auto_ok = self._set_enum(
            "GainAuto",
            gain_auto_value,
            "Continuous" if self.gain_auto else "Off",
        )

        if gain_auto_ok and not self.gain_auto:
            self._set_float("Gain", self.gain)

        # -------------- Exposure --------------
        if self.exposure_config_enable:
            exposure_auto_value = 2 if self.exposure_auto else 0
            exposure_auto_ok = self._set_enum(
                "ExposureAuto",
                exposure_auto_value,
                "Continuous" if self.exposure_auto else "Off",
            )

            if exposure_auto_ok and not self.exposure_auto:
                self._set_float("ExposureTime", self.exposure_time)

        # ---------------- Gamma ----------------
        if self.gamma_config_enable:
            gamma_enable_ok = self._set_bool(
                "GammaEnable",
                self.gamma_enable,
            )

            if gamma_enable_ok and self.gamma_enable:
                self._set_float("Gamma", self.gamma)

        # -------- Camera acquisition FPS -------
        # This differs from publish_fps:
        # publish_fps controls the ROS timer; camera_fps controls the device.
        if self.camera_fps_config_enable:
            fps_enable_ok = self._set_bool(
                "AcquisitionFrameRateEnable",
                True,
            )

            if fps_enable_ok:
                self._set_float(
                    "AcquisitionFrameRate",
                    self.camera_fps,
                )

    def _frame_to_cv(self, frame): # 帧转换为OpenCV图像
        info = frame.stFrameInfo
        width = int(info.nWidth) # 图像宽度
        height = int(info.nHeight) # 图像高度
        frame_len = int(info.nFrameLen) # 帧长度
        pixel_type = int(info.enPixelType) # 像素格式

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid frame size: "
                f"width={width}, height={height}"
            )

        if frame_len <= 0:
            raise RuntimeError(
                f"Empty camera frame: "
                f"width={width}, "
                f"height={height}, "
                f"frame_len={frame_len}"
            )

        if not frame.pBufAddr:
            raise RuntimeError(
                "Camera frame buffer address is NULL"
            )

        expected_size = width * height

        if frame_len < expected_size:
            raise RuntimeError(
                f"Incomplete Bayer frame: "
                f"frame_len={frame_len}, "
                f"expected={expected_size}"
            )

        PixelType_Gvsp_Mono8 = getattr(HIK, "PixelType_Gvsp_Mono8", None)
        PixelType_Gvsp_BGR8_Packed = getattr(HIK, "PixelType_Gvsp_BGR8_Packed", None)
        PixelType_Gvsp_RGB8_Packed = getattr(HIK, "PixelType_Gvsp_RGB8_Packed", None)
        PixelType_Gvsp_BayerRG8 = getattr(HIK,"PixelType_Gvsp_BayerRG8",None)

        if PixelType_Gvsp_Mono8 is not None and pixel_type == int(PixelType_Gvsp_Mono8):
            src = np.ctypeslib.as_array(
                cast(frame.pBufAddr, POINTER(c_ubyte)),
                shape=(frame_len,),
            ).copy()
            return src[: width * height].reshape(height, width)

        if ( PixelType_Gvsp_BGR8_Packed is not None
            and pixel_type == int(PixelType_Gvsp_BGR8_Packed)
        ):
            src = np.ctypeslib.as_array(
                cast(frame.pBufAddr, POINTER(c_ubyte)),
                shape=(frame_len,),
            ).copy()
            return src[: width * height * 3].reshape(height, width, 3)

        if ( PixelType_Gvsp_RGB8_Packed is not None
            and pixel_type == int(PixelType_Gvsp_RGB8_Packed)
        ):
            src = np.ctypeslib.as_array(
                cast(frame.pBufAddr, POINTER(c_ubyte)),
                shape=(frame_len,),
            ).copy()
            rgb = src[: width * height * 3].reshape(height, width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # BayerRG8
        if ( PixelType_Gvsp_BayerRG8 is not None
            and pixel_type == int(PixelType_Gvsp_BayerRG8)
        ):
            src = np.ctypeslib.as_array(
                cast(
                    frame.pBufAddr,
                    POINTER(c_ubyte)
                ),
                shape=(frame_len,),
            ).copy()

            bayer = src[: width * height].reshape(
                height,
                width
            )

            bgr = cv2.cvtColor(
                bayer,
                cv2.COLOR_BAYER_RG2BGR
            )

            return bgr

        # Generic SDK conversion to BGR8 for Bayer/other pixel formats.
        return self._convert_to_bgr8(frame)

    def _convert_to_bgr8(self, frame):
        info = frame.stFrameInfo
        width = int(info.nWidth)
        height = int(info.nHeight)

        dst_pixel = getattr(HIK, "PixelType_Gvsp_BGR8_Packed", None)
        if dst_pixel is None:
            raise RuntimeError(
                f"Unsupported pixel type 0x{int(info.enPixelType):x}; "
                "PixelType_Gvsp_BGR8_Packed not present in SDK."
            )

        Param = getattr(HIK, "MV_CC_PIXEL_CONVERT_PARAM_EX", None)
        if Param is None:
            Param = getattr(HIK, "MV_CC_PIXEL_CONVERT_PARAM", None)
        if Param is None:
            raise RuntimeError(
                "MVS SDK does not expose MV_CC_PIXEL_CONVERT_PARAM(_EX)."
            )

        param = Param()
        memset(byref(param), 0, sizeof(param))

        dst_size = width * height * 3
        dst = (c_ubyte * dst_size)()

        param.nWidth = width
        param.nHeight = height
        param.pSrcData = frame.pBufAddr
        param.nSrcDataLen = int(info.nFrameLen)
        param.enSrcPixelType = int(info.enPixelType)
        param.enDstPixelType = int(dst_pixel)
        param.pDstBuffer = dst
        param.nDstBufferSize = dst_size

        if hasattr(self.cam, "MV_CC_ConvertPixelTypeEx"):
            ret = self.cam.MV_CC_ConvertPixelTypeEx(param)
        elif hasattr(self.cam, "MV_CC_ConvertPixelType"):
            ret = self.cam.MV_CC_ConvertPixelType(param)
        else:
            raise RuntimeError("No pixel-conversion API found in MVS Python wrapper.")

        if ret != 0:
            raise RuntimeError(f"MVS pixel conversion failed: 0x{ret:08x}")

        actual_len = int(getattr(param, "nDstLen", dst_size))
        arr = np.ctypeslib.as_array(dst, shape=(dst_size,)).copy()
        arr = arr[: min(actual_len, dst_size)]
        if arr.size < dst_size:
            raise RuntimeError(
                f"Converted buffer too small: {arr.size} < {dst_size}"
            )
        return arr[:dst_size].reshape(height, width, 3)

    def _is_device_connected(self):
        """读取 MVS SDK 当前设备连接状态。"""
        if self.cam is None:
            return False

        if not hasattr(self.cam, "MV_CC_IsDeviceConnected"):
            return True

        try:
            return bool(
                self.cam.MV_CC_IsDeviceConnected()
            )
        except Exception:
            return False

    def _start_grabbing_once(self):
        """临时启动图像流。OpenDevice 必须已经保持打开。"""
        if self.cam is None or not self.device_opened:
            raise RuntimeError(
                "Camera device is not open."
            )

        if self.grabbing:
            return

        ret = self.cam.MV_CC_StartGrabbing()

        if ret != 0:
            raise RuntimeError(
                f"MV_CC_StartGrabbing failed: 0x{ret:08x}"
            )

        self.grabbing = True
        self.get_logger().info(
            "Run Once: MV_CC_StartGrabbing succeeded."
        )

    def _stop_grabbing_once(self):
        """
        停止本次图像流，但不 CloseDevice。
        相机仍保持长连接，等待下一次 Home 信号。
        """
        if (
            self.cam is None
            or not self.grabbing
        ):
            return

        try:
            ret = self.cam.MV_CC_StopGrabbing()

            if ret != 0:
                self.get_logger().warning(
                    f"MV_CC_StopGrabbing ret=0x{ret:08x}"
                )
            else:
                self.get_logger().info(
                    "Run Once: MV_CC_StopGrabbing succeeded; "
                    "camera remains OPEN."
                )
        finally:
            self.grabbing = False

    def _grab_one_frame(self, frame_index, total_frames):
        """
        从 MVS SDK 获取一帧并发布。

        每调用一次：
            GetImageBuffer
            -> 拷贝/转换图像
            -> 发布 ROS2 Image
            -> FreeImageBuffer

        返回：
            True  成功
            False 失败
        """
        MV_FRAME_OUT = self._symbol("MV_FRAME_OUT")

        frame = MV_FRAME_OUT()
        memset(
            byref(frame),
            0,
            sizeof(frame),
        )

        ret = self.cam.MV_CC_GetImageBuffer(
            frame,
            self.grab_timeout_ms,
        )

        if ret != 0:
            connected = self._is_device_connected()

            self.get_logger().warning(
                f"Burst frame {frame_index}/{total_frames} "
                f"GetImageBuffer failed: "
                f"0x{ret:08x}; "
                f"device_connected={connected}"
            )
            return False

        try:
            image = self._frame_to_cv(
                frame
            )

            if self.resize_scale != 1.0:
                image = cv2.resize(
                    image,
                    None,
                    fx=self.resize_scale,
                    fy=self.resize_scale,
                    interpolation=cv2.INTER_AREA,
                )

            if image.ndim == 2:
                msg = self.bridge.cv2_to_imgmsg(
                    image,
                    encoding="mono8",
                )
            else:
                msg = self.bridge.cv2_to_imgmsg(
                    image,
                    encoding="bgr8",
                )

            msg.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )
            msg.header.frame_id = self.frame_id

            self.pub.publish(
                msg
            )

            self.get_logger().info(
                f"Burst frame {frame_index}/{total_frames} "
                f"published: "
                f"{image.shape[1]}x{image.shape[0]} "
                f"-> {self.image_topic}"
            )

            return True

        except Exception as exc:
            self.get_logger().error(
                f"Burst frame {frame_index}/{total_frames} "
                f"processing failed: {exc}"
            )
            return False

        finally:
            # pBufAddr 属于 MVS SDK。
            # 每处理完一帧就立即归还 Buffer，再获取下一帧。
            try:
                self.cam.MV_CC_FreeImageBuffer(
                    frame
                )
            except Exception as exc:
                self.get_logger().warning(
                    f"Frame {frame_index}/{total_frames} "
                    f"FreeImageBuffer warning: {exc}"
                )


    def _run_once_callback(self, _msg):
        """
        收到机械臂 Home / Run Once 信号后执行一次完整视觉采集任务：

        已保持 OpenDevice
            -> StartGrabbing
            -> 连续 GetImageBuffer / FreeImageBuffer 共 burst_frames 次
            -> 每一帧立即发布 /hik_camera/image_raw
            -> StopGrabbing
            -> 继续保持 OpenDevice

        默认 burst_frames = 5。
        """
        if self.capture_busy:
            self.get_logger().warning(
                "Run Once request ignored: "
                "previous burst is still running."
            )
            return

        self.capture_busy = True

        successful_frames = 0
        success = False

        try:
            connected = self._is_device_connected()

            self.get_logger().info(
                "Run Once signal received. "
                f"device_connected={connected}; "
                f"starting {self.burst_frames}-frame burst."
            )

            if not connected:
                self.get_logger().error(
                    "Camera control connection is already lost "
                    "before StartGrabbing. "
                    "Burst cannot continue."
                )
                return

            # 一次 burst 只 StartGrabbing 一次。
            self._start_grabbing_once()

            # 连续从 SDK Buffer 取 burst_frames 帧。
            for frame_index in range(
                1,
                self.burst_frames + 1,
            ):
                frame_ok = self._grab_one_frame(
                    frame_index,
                    self.burst_frames,
                )

                if not frame_ok:
                    self.get_logger().error(
                        f"Burst aborted at frame "
                        f"{frame_index}/{self.burst_frames}."
                    )
                    break

                successful_frames += 1

            # 只有 5/5（或用户设置的 N/N）都成功才认为本次任务成功。
            success = (
                successful_frames
                == self.burst_frames
            )

        except Exception as exc:
            self.get_logger().error(
                f"Run Once burst failed: {exc}"
            )
            success = False

        finally:
            # Burst 结束只停止图像流，不关闭 OpenDevice。
            try:
                self._stop_grabbing_once()
            except Exception as exc:
                self.get_logger().warning(
                    f"Stop burst warning: {exc}"
                )

            done = Bool()
            done.data = bool(success)

            self.done_pub.publish(
                done
            )

            self.get_logger().info(
                f"Run Once burst finished: "
                f"{successful_frames}/{self.burst_frames} frames; "
                f"success={success}; "
                "camera remains OPEN for next Home signal."
            )

            self.capture_busy = False


    def destroy_node(self):
        """
        只有 ROS2 节点退出时，才真正关闭相机连接。
        """
        try:
            if self.cam is not None:
                if self.grabbing:
                    try:
                        self.cam.MV_CC_StopGrabbing()
                    except Exception:
                        pass

                self.grabbing = False

                if self.device_opened:
                    try:
                        ret = self.cam.MV_CC_CloseDevice()
                        if ret != 0:
                            self.get_logger().warning(
                                f"MV_CC_CloseDevice ret=0x{ret:08x}"
                            )
                    except Exception as exc:
                        self.get_logger().warning(
                            f"CloseDevice warning: {exc}"
                        )

                self.device_opened = False

                try:
                    ret = self.cam.MV_CC_DestroyHandle()
                    if ret != 0:
                        self.get_logger().warning(
                            f"MV_CC_DestroyHandle ret=0x{ret:08x}"
                        )
                except Exception as exc:
                    self.get_logger().warning(
                        f"DestroyHandle warning: {exc}"
                    )

                self.cam = None

            self.get_logger().info(
                "Camera connection closed because ROS2 node is shutting down."
            )

        except Exception as exc:
            self.get_logger().warning(
                f"Camera cleanup warning: {exc}"
            )

        super().destroy_node()

def main(args=None): # 主函数，初始化ROS2节点并运行
    rclpy.init(args=args)
    node = None
    try:
        node = HikCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[hik_camera_node] FATAL: {exc}", file=sys.stderr)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
