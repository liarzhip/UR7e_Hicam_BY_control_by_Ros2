#!/usr/bin/env python3
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# IMPORTANT:
# Training and ROS inference now share EXACTLY the same feature extractor,
# scaler and classifier implementation.
from ur7e_vision.traditional_svm import (
    SVMClassifier,
    extract_feature_vector,
    get_feature_dimension,
)


def normalize_rect_angle(rect):
    """
    Return an approximately [-90, 90) in-plane angle.
    OpenCV minAreaRect angle convention changes with rectangle side ordering.
    """
    (_, _), (w, h), angle = rect
    if w < h:
        angle += 90.0
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(angle)


def quaternion_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return q / n


def quaternion_multiply(q1, q2):
    """Quaternion order: [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def yaw_quaternion(yaw_rad):
    s = math.sin(yaw_rad / 2.0)
    c = math.cos(yaw_rad / 2.0)
    return np.array([0.0, 0.0, s, c], dtype=np.float64)


class SvmDetectorNode(Node):
    def __init__(self):
        super().__init__("svm_detector_node")

        share = Path(get_package_share_directory("ur7e_vision"))

        # ============================================================
        # ROS topics
        # ============================================================
        self.declare_parameter("image_topic", "/hik_camera/image_raw")
        self.declare_parameter("debug_topic", "/vision/debug_image")
        self.declare_parameter("target_pose_topic", "/vision/target_pose")
        self.declare_parameter("target_class_topic", "/vision/target_class")

        # ============================================================
        # SVM model parameters
        # ============================================================
        # Keep the old svm_model parameter for backward compatibility.
        # The original classifier needs FIVE files in the same directory:
        #   svm_model.xml
        #   svm_scaler.npz
        #   labels.json
        #   class_reference.npz
        #   model_meta.json
        self.declare_parameter(
            "svm_model",
            str(share / "models" / "svm_model.xml"),
        )

        # Optional explicit model directory.
        # If empty, the parent directory of svm_model is used.
        self.declare_parameter("svm_model_dir", str(share / "models"))
        self.declare_parameter("unknown_class_score_threshold", 0.15)

        # Keep labels_file as a compatibility/fallback interface.
        # When the complete SVMClassifier is loaded, labels.json from the
        # model directory is the authoritative class-name mapping.
        self.declare_parameter(
            "labels_file",
            str(share / "config" / "labels.yaml"),
        )

        self.declare_parameter(
            "homography_file",
            str(share / "calibration" / "homography.yaml"),
        )

        # ============================================================
        # Contour preprocessing parameters
        # ============================================================
        self.declare_parameter("min_area", 500.0)
        self.declare_parameter("max_area", 10000000.0)
        self.declare_parameter("blur_size", 5)
        self.declare_parameter("threshold_mode", "otsu")  # otsu | binary | canny
        self.declare_parameter("binary_threshold", 100)
        self.declare_parameter("canny_low", 50)
        self.declare_parameter("canny_high", 150)
        self.declare_parameter("invert_binary", False)

        # ============================================================
        # Robot target parameters
        # ============================================================
        self.declare_parameter("target_z", -999.0)
        self.declare_parameter("target_frame", "base")

        self.declare_parameter("fixed_qx", 0.995)
        self.declare_parameter("fixed_qy", 0.086)
        self.declare_parameter("fixed_qz", 0.019)
        self.declare_parameter("fixed_qw", -0.047)
        self.declare_parameter("use_detection_angle", False)
        self.declare_parameter("yaw_offset_deg", 0.0)

        self.declare_parameter("target_class_id", -1)
        self.declare_parameter("publish_when_no_svm", True)
        self.declare_parameter("stable_frames", 5)
        self.declare_parameter("stable_pixel_tolerance", 3.0)

        # ============================================================
        # Read parameters
        # ============================================================
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.target_class_topic = str(self.get_parameter("target_class_topic").value)

        svm_model_value = str(self.get_parameter("svm_model").value).strip()
        svm_model_dir_value = str(
            self.get_parameter("svm_model_dir").value
        ).strip()
        labels_value = str(self.get_parameter("labels_file").value).strip()
        homography_value = str(
            self.get_parameter("homography_file").value
        ).strip()

        self.svm_model_path = (
            Path(svm_model_value) if svm_model_value else None
        )

        if svm_model_dir_value:
            self.svm_model_dir = Path(svm_model_dir_value)
        elif self.svm_model_path is not None:
            self.svm_model_dir = self.svm_model_path.parent
        else:
            self.svm_model_dir = None

        self.unknown_class_score_threshold = float(
            self.get_parameter("unknown_class_score_threshold").value
        )

        self.labels_path = Path(labels_value) if labels_value else None
        self.homography_path = (
            Path(homography_value) if homography_value else None
        )

        self.min_area = float(self.get_parameter("min_area").value)
        self.max_area = float(self.get_parameter("max_area").value)
        self.blur_size = int(self.get_parameter("blur_size").value)
        self.threshold_mode = str(
            self.get_parameter("threshold_mode").value
        ).lower()
        self.binary_threshold = int(
            self.get_parameter("binary_threshold").value
        )
        self.canny_low = int(self.get_parameter("canny_low").value)
        self.canny_high = int(self.get_parameter("canny_high").value)
        self.invert_binary = bool(
            self.get_parameter("invert_binary").value
        )

        self.target_z = float(self.get_parameter("target_z").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.fixed_q = quaternion_normalize([
            float(self.get_parameter("fixed_qx").value),
            float(self.get_parameter("fixed_qy").value),
            float(self.get_parameter("fixed_qz").value),
            float(self.get_parameter("fixed_qw").value),
        ])
        self.use_detection_angle = bool(
            self.get_parameter("use_detection_angle").value
        )
        self.yaw_offset_deg = float(
            self.get_parameter("yaw_offset_deg").value
        )

        self.target_class_id = int(
            self.get_parameter("target_class_id").value
        )
        self.publish_when_no_svm = bool(
            self.get_parameter("publish_when_no_svm").value
        )
        self.stable_frames = max(
            1,
            int(self.get_parameter("stable_frames").value),
        )
        self.stable_pixel_tolerance = float(
            self.get_parameter("stable_pixel_tolerance").value
        )

        # ============================================================
        # Runtime objects
        # ============================================================
        self.bridge = CvBridge()

        # Complete original SVM pipeline:
        # feature extraction -> scaler -> SVM -> labels/reference score.
        self.classifier = self._load_classifier()

        # Compatibility fallback only when classifier is unavailable.
        self.labels = self._load_labels()

        self.H = self._load_homography()

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.target_pose_topic,
            10,
        )
        self.class_pub = self.create_publisher(
            String,
            self.target_class_topic,
            10,
        )
        self.debug_pub = self.create_publisher(
            Image,
            self.debug_topic,
            5,
        )
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.on_image,
            5,
        )

        self.stable_history = []

        self.get_logger().info(
            f"Detector ready. image={self.image_topic}, "
            f"target={self.target_pose_topic}"
        )

        if self.classifier is not None:
            self.get_logger().info(
                "SVM pipeline ready: "
                f"feature_dimension={get_feature_dimension()}, "
                f"model_dir={self.svm_model_dir}"
            )

        if self.H is None:
            self.get_logger().warning(
                "No valid homography loaded. Detection/SVM classification "
                "will work, but target_pose will NOT be published until "
                "planar calibration is completed."
            )

        if self.target_z < -10.0:
            self.get_logger().warning(
                "target_z is not configured. target_pose publication is disabled."
            )

    # ================================================================
    # Model / calibration loading
    # ================================================================
    def _load_classifier(self):
        if self.svm_model_dir is None:
            self.get_logger().warning(
                "SVM model directory is not configured. "
                "Running contour-only mode."
            )
            return None

        try:
            classifier = SVMClassifier(
                self.svm_model_dir,
                unknown_score_threshold=(
                    self.unknown_class_score_threshold
                ),
            )
        except Exception as exc:
            self.get_logger().error(
                "Failed to load complete SVM pipeline from "
                f"{self.svm_model_dir}: {exc}"
            )
            self.get_logger().warning(
                "SVM classification is disabled. "
                "Make sure svm_model.xml, svm_scaler.npz, labels.json, "
                "class_reference.npz and model_meta.json are all in the "
                "same model directory."
            )
            return None

        model_feature_count = int(classifier.svm.getVarCount())
        extractor_feature_count = int(get_feature_dimension())

        if model_feature_count != extractor_feature_count:
            raise RuntimeError(
                "SVM feature dimension mismatch even after loading the "
                "original training pipeline: "
                f"model={model_feature_count}, "
                f"extractor={extractor_feature_count}."
            )

        self.get_logger().info(
            "Loaded complete SVM pipeline: "
            f"{self.svm_model_dir}, "
            f"features={model_feature_count}"
        )
        return classifier

    def _load_labels(self):
        if self.labels_path is None or not self.labels_path.is_file():
            return {}

        with open(
            self.labels_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f) or {}

        raw = data.get("labels", {})
        return {
            int(k): str(v)
            for k, v in raw.items()
        }

    def _load_homography(self):
        if (
            self.homography_path is None
            or not self.homography_path.is_file()
        ):
            return None

        with open(
            self.homography_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f) or {}

        values = data.get("homography", {}).get("data", None)

        if values is None or len(values) != 9:
            self.get_logger().error(
                f"Invalid homography file: {self.homography_path}"
            )
            return None

        H = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(3, 3)

        self.get_logger().info(
            f"Loaded homography: {self.homography_path}"
        )
        return H

    # ================================================================
    # Image preprocessing / SVM
    # ================================================================
    def preprocess(self, gray):
        k = self.blur_size

        if k < 1:
            k = 1

        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(
            gray,
            (k, k),
            0,
        )

        if self.threshold_mode == "canny":
            return cv2.Canny(
                blurred,
                self.canny_low,
                self.canny_high,
            )

        if self.threshold_mode == "binary":
            flag = (
                cv2.THRESH_BINARY_INV
                if self.invert_binary
                else cv2.THRESH_BINARY
            )
            _, mask = cv2.threshold(
                blurred,
                self.binary_threshold,
                255,
                flag,
            )
            return mask

        flag = (
            cv2.THRESH_BINARY_INV
            if self.invert_binary
            else cv2.THRESH_BINARY
        )

        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            flag | cv2.THRESH_OTSU,
        )

        return mask

    def classify(self, frame, contour):
        """
        Run the EXACT feature pipeline used by the original training code.

        Returns:
            (class_name, class_score, class_id)

        If the SVM is unavailable:
            ("unclassified", 0.0, -1)
        """
        if self.classifier is None:
            return "unclassified", 0.0, -1

        feature_vector, normalized_patch, feature_details = (
            extract_feature_vector(
                frame,
                contour,
            )
        )

        expected = int(get_feature_dimension())

        if feature_vector.size != expected:
            raise RuntimeError(
                "Feature extractor returned unexpected dimension: "
                f"{feature_vector.size}, expected {expected}."
            )

        class_name, class_score, class_id = (
            self.classifier.predict(
                feature_vector
            )
        )

        return (
            str(class_name),
            float(class_score),
            int(class_id),
        )

    # ================================================================
    # Coordinate / stability helpers
    # ================================================================
    def pixel_to_base_xy(self, u, v):
        src = np.array(
            [[[float(u), float(v)]]],
            dtype=np.float64,
        )
        dst = cv2.perspectiveTransform(
            src,
            self.H,
        )
        return (
            float(dst[0, 0, 0]),
            float(dst[0, 0, 1]),
        )

    def is_stable(self, u, v):
        self.stable_history.append(
            np.array(
                [u, v],
                dtype=np.float64,
            )
        )

        if len(self.stable_history) > self.stable_frames:
            self.stable_history.pop(0)

        if len(self.stable_history) < self.stable_frames:
            return False

        arr = np.vstack(
            self.stable_history
        )
        center = arr.mean(
            axis=0
        )
        max_dev = np.linalg.norm(
            arr - center,
            axis=1,
        ).max()

        return bool(
            max_dev <= self.stable_pixel_tolerance
        )

    # ================================================================
    # Main image callback
    # ================================================================
    def on_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough",
            )

            if img.ndim == 2:
                gray = img
                debug = cv2.cvtColor(
                    gray,
                    cv2.COLOR_GRAY2BGR,
                )
            else:
                debug = img.copy()
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY,
                )

            mask = self.preprocess(
                gray
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            candidates = []

            for contour in contours:
                area = float(
                    cv2.contourArea(contour)
                )

                if (
                    area < self.min_area
                    or area > self.max_area
                ):
                    continue

                rect = cv2.minAreaRect(
                    contour
                )
                (u, v), (rw, rh), _ = rect

                if rw <= 1.0 or rh <= 1.0:
                    continue

                try:
                    (
                        class_name,
                        class_score,
                        class_id,
                    ) = self.classify(
                        img,
                        contour,
                    )
                except Exception as exc:
                    self.get_logger().error(
                        f"SVM classification failed: {exc}"
                    )
                    continue

                if (
                    self.classifier is not None
                    and self.target_class_id >= 0
                    and class_id != self.target_class_id
                ):
                    continue

                if (
                    self.classifier is None
                    and not self.publish_when_no_svm
                ):
                    continue

                angle_deg = normalize_rect_angle(
                    rect
                )

                candidates.append(
                    (
                        area,
                        contour,
                        u,
                        v,
                        angle_deg,
                        class_name,
                        class_score,
                        class_id,
                    )
                )

            if not candidates:
                self.stable_history.clear()
                self._publish_debug(
                    msg,
                    debug,
                )
                return

            # Preserve the original ROS V1 policy:
            # choose the largest accepted contour.
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            (
                area,
                contour,
                u,
                v,
                angle_deg,
                class_name,
                class_score,
                class_id,
            ) = candidates[0]

            box = cv2.boxPoints(
                cv2.minAreaRect(contour)
            ).astype(np.int32)

            cv2.drawContours(
                debug,
                [box],
                0,
                (0, 255, 0),
                2,
            )

            cv2.circle(
                debug,
                (
                    int(round(u)),
                    int(round(v)),
                ),
                5,
                (0, 0, 255),
                -1,
            )

            # When the complete classifier is available, labels.json from the
            # original model is authoritative.  labels.yaml is only a fallback
            # for contour-only mode / legacy use.
            if self.classifier is not None:
                label = class_name
            else:
                label = self.labels.get(
                    class_id,
                    f"class_{class_id}",
                )

            cv2.putText(
                debug,
                (
                    f"{label} "
                    f"score={class_score:.3f} "
                    f"u={u:.1f} v={v:.1f} "
                    f"a={angle_deg:.1f}"
                ),
                (
                    max(0, int(u) - 180),
                    max(25, int(v) - 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            stable = self.is_stable(
                u,
                v,
            )

            cv2.putText(
                debug,
                f"stable={stable}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            class_msg = String()
            class_msg.data = label
            self.class_pub.publish(
                class_msg
            )

            # Never publish a robot target from:
            #   1) an unstable detection,
            #   2) uncalibrated image coordinates,
            #   3) an unconfigured target Z,
            #   4) an SVM result rejected as "unknown".
            if (
                stable
                and self.H is not None
                and self.target_z > -10.0
                and label != "unknown"
            ):
                base_x, base_y = self.pixel_to_base_xy(
                    u,
                    v,
                )

                q = self.fixed_q.copy()

                if self.use_detection_angle:
                    yaw = math.radians(
                        angle_deg
                        + self.yaw_offset_deg
                    )
                    q = quaternion_multiply(
                        yaw_quaternion(yaw),
                        q,
                    )
                    q = quaternion_normalize(
                        q
                    )

                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = self.target_frame

                pose.pose.position.x = base_x
                pose.pose.position.y = base_y
                pose.pose.position.z = self.target_z

                pose.pose.orientation.x = float(q[0])
                pose.pose.orientation.y = float(q[1])
                pose.pose.orientation.z = float(q[2])
                pose.pose.orientation.w = float(q[3])

                self.pose_pub.publish(
                    pose
                )

                cv2.putText(
                    debug,
                    (
                        f"base=({base_x:.3f},"
                        f"{base_y:.3f},"
                        f"{self.target_z:.3f})"
                    ),
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            self._publish_debug(
                msg,
                debug,
            )

        except Exception as exc:
            self.get_logger().error(
                f"Detector callback failed: {exc}"
            )

    def _publish_debug(self, source_msg, debug):
        out = self.bridge.cv2_to_imgmsg(
            debug,
            encoding="bgr8",
        )
        out.header = source_msg.header
        self.debug_pub.publish(
            out
        )


def main(args=None):
    rclpy.init(args=args)
    node = SvmDetectorNode()

    try:
        rclpy.spin(
            node
        )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
