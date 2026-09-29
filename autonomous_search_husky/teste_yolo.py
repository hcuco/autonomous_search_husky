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

# Novos imports para a matemática 3D e Transformações (TF)
from tf2_ros import Buffer, TransformListener
import tf2_ros

class PrimeiroTesteNode(Node):

    def __init__(self):
        super().__init__("teste_subscriber")

        # Configuração de QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # Mantido em BEST_EFFORT para simulação
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.rgb_subscriber_ = Subscriber(
            self, Image, "/a200_1077/sensors/camera_0/color/image", qos_profile=sensor_qos
        )

        self.depth_subscriber_ = Subscriber(
            self, Image, "/a200_1077/sensors/camera_0/depth/image", qos_profile=sensor_qos
        )

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

        # Buffer e Listener de TF para descobrir a distância da Câmera até a Base do Robô
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLO prediction model
        self.model = YOLO("yolov8n.pt")
        self.bridge = CvBridge()

        # Publicador Reliability para o RViz
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.publisher_ = self.create_publisher(Image, "/a200_1077/sensors/camera_YOLO", pub_qos)

        print("Classes detectáveis:", self.model.names) 

        class_dict = {value: key for key, value in self.model.names.items()}
        self.person_id = class_dict.get('person')
        self.chair_id = class_dict.get('chair')

        print(f"ID Pessoa: {self.person_id}, ID Cadeira: {self.chair_id}")

    # Função auxiliar matemática para converter Quaternions (da TF) para Matriz de Rotação
    def quat_to_mat(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    def callback_camera_info(self, msg: CameraInfo):
        if self.camera_info is None:
            self.get_logger().info("CameraInfo recebido e travado.")
        self.camera_info = msg

    def callback_imagens(self, rgb_msg: Image, depth_msg: Image):

        if self.camera_info is None:
            self.get_logger().warning("Aguardando CameraInfo...")
            return

        try:
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

                u = max(0, min(u, depth_image.shape[1] - 1))
                v = max(0, min(v, depth_image.shape[0] - 1))

                Z = float(depth_image[v, u])

                if depth_msg.encoding == "16UC1":
                    Z /= 1000.0  

                if not np.isfinite(Z) or Z <= 0.0:
                    continue

                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx = self.camera_info.k[2]
                cy = self.camera_info.k[5]

                # 1. Matemática da Lente (Frame Óptico)
                X_opt = (u - cx) * Z / fx
                Y_opt = (v - cy) * Z / fy
                Z_opt = Z

                # 2. Converte do padrão da lente para o padrão do Robô (camera_0_link)
                # Na lente: Z é para frente. No robô: X é para frente.
                x_cam = Z_opt
                y_cam = -X_opt
                z_cam = -Y_opt

                # 3. Pede ao ROS a distância física entre a câmera e o centro do robô
                try:
                    # Se base_link falhar, altere para "a200_1077/robot/base_link" ou "odom"
                    trans = self.tf_buffer.lookup_transform(
                        "base_link",          # Quero a posição baseada no centro do robô
                        "camera_0_link",      # A partir da câmera
                        rclpy.time.Time()
                    )
                except Exception as e:
                    self.get_logger().warning(f"Aguardando a árvore de TF do robô: {e}")
                    continue

                # 4. Projeta as coordenadas para o chão do robô usando matrizes
                t = trans.transform.translation
                R = self.quat_to_mat(trans.transform.rotation)

                point_cam = np.array([x_cam, y_cam, z_cam])
                point_base = R.dot(point_cam) + np.array([t.x, t.y, t.z])

                X_base, Y_base, Z_base = point_base
                distancia_2d_chao = np.sqrt(X_base**2 + Y_base**2)

                self.get_logger().info(
                    f"\n--- {class_name.upper()} DETECTADO ---\n"
                    f"Distância real (no chão): {distancia_2d_chao:.2f} m\n"
                    f"Coordenadas de Navegação: X(Frente)={X_base:.2f}m | Y(Lado)={Y_base:.2f}m\n"
                    f"----------------------------"
                )
            
            # Imagem processada em array numpy (BGR)
            annotated_frame = result.plot()

            # Bypass do cv_bridge para o YOLO
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