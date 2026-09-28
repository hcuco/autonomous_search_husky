from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import traceback

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from cv_bridge import CvBridge

class PrimeiroTesteNode(Node):

    def __init__(self):
        super().__init__("teste_subscriber")

        # qos configuration to synchronize depth and rgb messages
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Atualize os tópicos se necessário para bater com o seu robô/simulação
        self.rgb_subscriber_ = Subscriber(
            self, Image, "/a200_1077/sensors/camera_0/color/image", qos_profile=sensor_qos
        )

        self.depth_subscriber_ = Subscriber(
            self, Image, "/a200_1077/sensors/camera_0/depth/image", qos_profile=sensor_qos
        )

        # sync between rgb and depth
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.rgb_subscriber_, self.depth_subscriber_],
            queue_size=5,
            slop=0.05,
        )

        self.synchronizer.registerCallback(self.callback_imagens)

        self.camera_info = None

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/a200_1077/sensors/camera_0/color/camera_info",
            self.callback_camera_info,
            sensor_qos,
        )

        # YOLO prediction model - usando YOLOv8 nano pré-treinado no COCO
        self.model = YOLO("yolov8n.pt")
        self.bridge = CvBridge()

        self.publisher_ = self.create_publisher(Image, "/a200_1077/sensors/camera_YOLO", 10)

        # Print the entire dictionary of names (Opcional, bom para debug)
        print("Classes detectáveis:", self.model.names) 

        # Dynamically find the class IDs for 'person' and 'chair'
        class_dict = {value: key for key, value in self.model.names.items()}
        self.person_id = class_dict.get('person')
        self.chair_id = class_dict.get('chair')

        print(f"ID Pessoa: {self.person_id}, ID Cadeira: {self.chair_id}")

    def callback_camera_info(self, msg: CameraInfo):
        if self.camera_info is None:
            self.get_logger().info(
                f"CameraInfo recebido: "
                f"{msg.width}x{msg.height}, "
                f"frame={msg.header.frame_id}, "
                f"fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}"
            )
        self.camera_info = msg

    def callback_imagens(self, rgb_msg: Image, depth_msg: Image):

        if self.camera_info is None:
            self.get_logger().warning("Aguardando CameraInfo...")
            return

        try:
            # Converte as mensagens do ROS 2 para OpenCV
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            results = self.model.predict(
                source=rgb_image, 
                show=False, 
                conf=0.55, 
                classes=[self.person_id, self.chair_id], 
                verbose=False
            )

            result = results[0]

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                u = int(round((x1 + x2) / 2))
                v = int(round((y1 + y2) / 2))

                # Impede acesso fora da imagem
                u = max(0, min(u, depth_image.shape[1] - 1))
                v = max(0, min(v, depth_image.shape[0] - 1))

                # Obtém a profundidade crua
                Z = float(depth_image[v, u])

                # Ajuste de escala baseado no formato da imagem (Milímetros vs Metros)
                if depth_msg.encoding == "16UC1":
                    Z /= 1000.0  # Converte mm para metros

                if not np.isfinite(Z) or Z <= 0.0:
                    self.get_logger().warning(
                        f"{class_name}: profundidade inválida no pixel ({u}, {v}) - Z: {Z}"
                    )
                    continue

                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx = self.camera_info.k[2]
                cy = self.camera_info.k[5]

                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                # Posição 3D da vítima em relação à lente da câmera
                point_camera = (X, Y, Z)

                self.get_logger().info(
                    f"{class_name}: "
                    f"pixel=({u}, {v}), "
                    f"depth={Z:.3f} m, "
                    f"P_camera=({X:.3f}, {Y:.3f}, {Z:.3f}) m, "
                    f"confiança={confidence:.2f}"
                )
            
            # Imagem processada em array numpy (BGR)
            annotated_frame = result.plot()

            # Bypass do cv_bridge: Montando a mensagem Image manualmente para evitar o KeyError: 16
            output_msg = Image()
            output_msg.header = rgb_msg.header
            output_msg.height = annotated_frame.shape[0]
            output_msg.width = annotated_frame.shape[1]
            output_msg.encoding = 'bgr8'
            output_msg.is_bigendian = 0
            output_msg.step = annotated_frame.shape[1] * 3
            output_msg.data = annotated_frame.tobytes()

            self.publisher_.publish(output_msg)

        except Exception as e:
            # Imprime o erro com rastreamento completo da linha exata onde ocorreu
            self.get_logger().error(f"Erro ao processar imagem: {e}\n{traceback.format_exc()}")
            
def main(args=None):
    rclpy.init(args=args)
    node = PrimeiroTesteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()