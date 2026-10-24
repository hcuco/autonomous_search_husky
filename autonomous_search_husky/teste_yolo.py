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

# Imports para a matemática 3D e Transformações (TF)
from tf2_ros import Buffer, TransformListener
import tf2_ros
import cv2

class PrimeiroTesteNode(Node):

    def __init__(self):
        super().__init__("teste_subscriber")

        # 1 e 2. Declaração de Parâmetros (Namespace, Frames e Modelo YOLO)
        self.declare_parameter("yolo_model", "yolov8n-pose.pt")
        self.declare_parameter("robot_namespace", "/a200_1077")
        self.declare_parameter("global_frame", "odom") # Pode ser 'map' ou 'odom'
        self.declare_parameter("base_frame", "base_link") # Se falhar, use 'a200_1077/base_link'

        self.yolo_model_name = self.get_parameter("yolo_model").value
        ns = self.get_parameter("robot_namespace").value
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        # Tratamento simples para garantir formatação correta do namespace
        if ns and not ns.startswith('/'):
            ns = '/' + ns
        if ns.endswith('/'):
            ns = ns[:-1]

        # Configuração de QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Assinaturas utilizando o namespace dinâmico
        self.rgb_subscriber_ = Subscriber(
            self, Image, f"{ns}/sensors/camera_0/color/image", qos_profile=sensor_qos
        )

        self.depth_subscriber_ = Subscriber(
            self, Image, f"{ns}/sensors/camera_0/depth/image", qos_profile=sensor_qos
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
            f"{ns}/sensors/camera_0/color/camera_info",
            self.callback_camera_info,
            sensor_qos,
        )

        # Buffer de TF
        self.tf_buffer = Buffer()
        
        # Mini-nó oculto exclusivo para o TF. Isso elimina a necessidade de digitar --remap no terminal
        tf_node_args = ['--ros-args', '--remap', f'/tf:={ns}/tf', '--remap', f'/tf_static:={ns}/tf_static']
        self.tf_node = rclpy.create_node('yolo_tf_listener', use_global_arguments=False, cli_args=tf_node_args)
        
        # O spin_thread=True fará com que o TransformListener rode em background de forma segura
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)

        # YOLO load dinâmico
        self.get_logger().info(f"Carregando modelo YOLO: {self.yolo_model_name}")
        self.model = YOLO(self.yolo_model_name)
        self.bridge = CvBridge()

        # Busca dinâmica pela classe 'person' (evita quebrar caso troque de modelo)
        class_dict = {value: key for key, value in self.model.names.items()}
        self.person_id = class_dict.get('person')
        
        if self.person_id is None:
            self.get_logger().error(f"O modelo {self.yolo_model_name} não possui a classe 'person'.")
        else:
            self.get_logger().info(f"ID Pessoa encontrado: {self.person_id}")

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.publisher_ = self.create_publisher(Image, f"{ns}/sensors/camera_YOLO", pub_qos)

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
        if self.camera_info is None or self.person_id is None:
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            results = self.model.predict(
                source=rgb_image, 
                show=False, 
                conf=0.55, 
                classes=[self.person_id], 
                verbose=False
            )

            result = results[0]

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                class_name = "person"

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

                # Matemática da lente
                X_opt = (u - cx) * Z / fx
                Y_opt = (v - cy) * Z / fy
                Z_opt = Z

                # Câmera para padrão ROS
                x_cam = Z_opt
                y_cam = -X_opt
                z_cam = -Y_opt
                point_cam = np.array([x_cam, y_cam, z_cam])

                # 3. Matemática Dupla de TF: Relativa (Robô) e Global (Odom/Map)
                try:
                    # 3.1 Transf. Relativa (da câmera para o centro do robô)
                    trans_base = self.tf_buffer.lookup_transform(
                        self.base_frame,
                        "camera_0_link",
                        rclpy.time.Time()
                    )
                    # 3.2 Transf. Global (da câmera para o mapa/odom global)
                    trans_global = self.tf_buffer.lookup_transform(
                        self.global_frame,
                        "camera_0_link",
                        rclpy.time.Time()
                    )
                except Exception as e:
                    self.get_logger().warning(f"Aguardando a árvore de TF completa: {e}")
                    continue

                # --- Cálculo Relativo ---
                t_b = trans_base.transform.translation
                R_b = self.quat_to_mat(trans_base.transform.rotation)
                point_base = R_b.dot(point_cam) + np.array([t_b.x, t_b.y, t_b.z])
                X_base, Y_base, Z_base = point_base
                distancia_2d_chao = np.sqrt(X_base**2 + Y_base**2)

                # --- Cálculo Global ---
                t_g = trans_global.transform.translation
                R_g = self.quat_to_mat(trans_global.transform.rotation)
                point_global = R_g.dot(point_cam) + np.array([t_g.x, t_g.y, t_g.z])
                X_glob, Y_glob, Z_glob = point_global

                self.get_logger().info(
                    f"\n--- {class_name.upper()} DETECTADO ---\n"
                    f"Distância real do robô: {distancia_2d_chao:.2f} m\n"
                    f"[Relativo] X(Frente)={X_base:.2f}m | Y(Lado)={Y_base:.2f}m\n"
                    f"[Global] X({self.global_frame})={X_glob:.2f}m | Y({self.global_frame})={Y_glob:.2f}m\n"
                    f"----------------------------"
                )
            
            annotated_frame = result.plot()

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
        # Destrói o mini-nó do TF para liberar a thread
        if hasattr(node, 'tf_node'):
            node.tf_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()