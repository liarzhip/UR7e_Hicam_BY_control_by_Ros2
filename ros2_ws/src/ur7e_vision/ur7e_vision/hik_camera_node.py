#!/usr/bin/env python3
import os
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


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
        module = __import__("MvCameraControl_class", fromlist=["*"])
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
        self.declare_parameter("publish_fps", 15.0)
        self.declare_parameter("frame_id", "hik_camera_optical_frame")
        self.declare_parameter("image_topic", "/hik_camera/image_raw")
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

        # False = continuous acquisition; True = trigger mode.
        self.declare_parameter("trigger_mode", False)

        self.device_index = int(self.get_parameter("device_index").value)
        self.publish_fps = float(self.get_parameter("publish_fps").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
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

        self.trigger_mode = bool(self.get_parameter("trigger_mode").value)


        if self.publish_fps <= 0.0:
            raise ValueError("publish_fps must be > 0")
        if self.resize_scale <= 0.0:
            raise ValueError("resize_scale must be > 0")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.image_topic, 5)

        self.cam = None
        self._open_camera()

        self.timer = self.create_timer(1.0 / self.publish_fps, self._grab_once)
        self.get_logger().info(
            f"HIK camera ready: topic={self.image_topic}, "
            f"fps={self.publish_fps:.1f}, frame_id={self.frame_id}"
        )

    @staticmethod
    def _symbol(name):
        if not hasattr(HIK, name):
            raise RuntimeError(f"HIK SDK symbol missing: {name}")
        return getattr(HIK, name)

    def _enumerate_devices(self):
        MV_CC_DEVICE_INFO_LIST = self._symbol("MV_CC_DEVICE_INFO_LIST")
        MV_GIGE_DEVICE = self._symbol("MV_GIGE_DEVICE")
        MV_USB_DEVICE = self._symbol("MV_USB_DEVICE")
        MvCamera = self._symbol("MvCamera")

        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer, device_list)
        if ret != 0:
            raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08x}")

        if device_list.nDeviceNum == 0:
            raise RuntimeError("No HIKROBOT camera found by MVS SDK.")

        self.get_logger().info(f"Found {device_list.nDeviceNum} HIKROBOT camera(s).")
        return device_list

    def _open_camera(self):
        MvCamera = self._symbol("MvCamera")
        MV_CC_DEVICE_INFO = self._symbol("MV_CC_DEVICE_INFO")
        MV_ACCESS_Exclusive = self._symbol("MV_ACCESS_Exclusive")

        device_list = self._enumerate_devices()
        if self.device_index < 0 or self.device_index >= device_list.nDeviceNum:
            raise RuntimeError(
                f"device_index={self.device_index} out of range "
                f"[0, {device_list.nDeviceNum - 1}]"
            )

        device_info = cast(
            device_list.pDeviceInfo[self.device_index],
            POINTER(MV_CC_DEVICE_INFO),
        ).contents

        self.cam = MvCamera()

        ret = self.cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:08x}")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:08x}")

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

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:08x}")

    def _set_enum(self, name, value, label=None):
        """Safely set an MVS enum node."""
        ret = self.cam.MV_CC_SetEnumValue(name, int(value))
        shown = label if label is not None else str(value)
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set {name}={shown}, ret=0x{ret:08x}"
            )
            return False

        self.get_logger().info(f"{name} = {shown}")
        return True

    def _set_float(self, name, value):
        """Safely set an MVS floating-point node."""
        ret = self.cam.MV_CC_SetFloatValue(name, float(value))
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set {name}={value}, ret=0x{ret:08x}"
            )
            return False

        self.get_logger().info(f"{name} = {float(value):.3f}")
        return True

    def _set_bool(self, name, value):
        """Safely set an MVS boolean node when supported by the SDK."""
        if not hasattr(self.cam, "MV_CC_SetBoolValue"):
            self.get_logger().warning(
                f"SDK has no MV_CC_SetBoolValue; skip {name}."
            )
            return False

        ret = self.cam.MV_CC_SetBoolValue(name, bool(value))
        if ret != 0:
            self.get_logger().warning(
                f"Failed to set {name}={bool(value)}, ret=0x{ret:08x}"
            )
            return False

        self.get_logger().info(f"{name} = {bool(value)}")
        return True

    def _configure_camera_parameters(self):
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

    def _frame_to_cv(self, frame):
        info = frame.stFrameInfo
        width = int(info.nWidth)
        height = int(info.nHeight)
        frame_len = int(info.nFrameLen)
        pixel_type = int(info.enPixelType)

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

        if (
            PixelType_Gvsp_BGR8_Packed is not None
            and pixel_type == int(PixelType_Gvsp_BGR8_Packed)
        ):
            src = np.ctypeslib.as_array(
                cast(frame.pBufAddr, POINTER(c_ubyte)),
                shape=(frame_len,),
            ).copy()
            return src[: width * height * 3].reshape(height, width, 3)

        if (
            PixelType_Gvsp_RGB8_Packed is not None
            and pixel_type == int(PixelType_Gvsp_RGB8_Packed)
        ):
            src = np.ctypeslib.as_array(
                cast(frame.pBufAddr, POINTER(c_ubyte)),
                shape=(frame_len,),
            ).copy()
            rgb = src[: width * height * 3].reshape(height, width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # BayerRG8
        if (
            PixelType_Gvsp_BayerRG8 is not None
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

    def _grab_once(self):
        MV_FRAME_OUT = self._symbol("MV_FRAME_OUT")
        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))

        ret = self.cam.MV_CC_GetImageBuffer(frame, 1000)
        if ret != 0:
            self.get_logger().warning(
                f"MV_CC_GetImageBuffer timeout/error: 0x{ret:08x}",
                throttle_duration_sec=2.0,
            )
            return

        try:
            image = self._frame_to_cv(frame)

            if self.resize_scale != 1.0:
                image = cv2.resize(
                    image,
                    None,
                    fx=self.resize_scale,
                    fy=self.resize_scale,
                    interpolation=cv2.INTER_AREA,
                )

            if image.ndim == 2:
                msg = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
            else:
                msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.pub.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"Frame processing failed: {exc}")
        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)

    def destroy_node(self):
        try:
            if self.cam is not None:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
        except Exception as exc:
            self.get_logger().warning(f"Camera cleanup warning: {exc}")
        super().destroy_node()


def main(args=None):
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
