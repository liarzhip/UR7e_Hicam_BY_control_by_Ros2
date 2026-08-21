import pyrealsense2 as rs
import numpy as np
import cv2

# =========================
# RealSense初始化
# =========================
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)
config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

profile = pipeline.start(config)

# =========================
# 深度和彩色对齐
# =========================
align = rs.align(
    rs.stream.color
)


# =========================
# 获取相机内参
# =========================
depth_profile = profile.get_stream(
    rs.stream.depth
)

intr = depth_profile.as_video_stream_profile().intrinsics


print("Camera intrinsics:")
print("fx =", intr.fx)
print("fy =", intr.fy)
print("cx =", intr.ppx)
print("cy =", intr.ppy)

# 保存点击坐标
click_point = None


# =========================
# 鼠标回调函数
# =========================
def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

# 创建窗口
cv2.namedWindow(
    "RealSense RGB"
)

cv2.setMouseCallback(
    "RealSense RGB",
    mouse_callback
)

try:
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        # 对齐Depth到RGB
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 转numpy
        color_image = np.asanyarray(
            color_frame.get_data()
        )
        # 如果点击了
        if click_point is not None:
            u,v = click_point
            # 获取深度(m)
            depth = depth_frame.get_distance(
                u,
                v
            )
            if depth == 0:
                print("Invalid depth")
            else:
                # 像素+深度
                pixel = [
                    u,
                    v
                ]
                # 反投影到3D
                point = rs.rs2_deproject_pixel_to_point(
                    intr,
                    pixel,
                    depth
                )

                X,Y,Z = point

                print(
                    "Camera Coordinate:"
                )
                print(
                    "X = %.3f m" % X
                )
                print(
                    "Y = %.3f m" % Y
                )
                print(
                    "D = %.3f m" % Z
                )

            # 画点击点
            cv2.circle(
                color_image,
                (u,v),
                5,
                (0,0,255),
                -1
            )

            click_point = None

        cv2.imshow(
            "RealSense RGB",
            color_image
        )

        key=cv2.waitKey(1)

        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()