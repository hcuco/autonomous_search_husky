import math

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import traceback
from autonomous_search_msgs.msg import VictimDetection # <--- Import da sua nova msg

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

# exemplo de comando para rodar este nó com parâmetros específicos:
# ros2 run autonomous_search_husky teste_yolo --ros-args -p use_sim_time:=true -p yolo_model:=yolov8n.pt -p global_frame:=map -p robot_namespace:=/a200_1077

class PrimeiroTesteNode(Node):

    def __init__(self):
        super().__init__("teste_subscriber")

        # Parâmetros
        self.declare_parameter("yolo_model", "yolov8n-pose.pt")
        self.declare_parameter("robot_namespace", "/a200_1077")
        self.declare_parameter("global_frame", "odom") 
        self.declare_parameter("base_frame", "base_link") 

        self.yolo_model_name = self.get_parameter("yolo_model").value
        ns = self.get_parameter("robot_namespace").value
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        if ns and not ns.startswith('/'): ns = '/' + ns
        if ns.endswith('/'): ns = ns[:-1]

        # Configuração de QoS para os sensores
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.rgb_subscriber_ = Subscriber(
            self, Image, f"{ns}/sensors/camera_0/color/image", qos_profile=sensor_qos
        )
        self.depth_subscriber_ = Subscriber(
            self, Image, f"{ns}/sensors/camera_0/depth/image", qos_profile=sensor_qos
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [self.rgb_subscriber_, self.depth_subscriber_], queue_size=5, slop=0.05
        )
        self.synchronizer.registerCallback(self.callback_imagens)

        self.camera_info = None
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo, f"{ns}/sensors/camera_0/color/camera_info", self.callback_camera_info, sensor_qos
        )

        # Buffer de TF com Nó auxiliar para evitar necessidade de remap no terminal
        self.tf_buffer = Buffer()
        tf_node_args = ['--ros-args', '--remap', f'/tf:={ns}/tf', '--remap', f'/tf_static:={ns}/tf_static']
        self.tf_node = rclpy.create_node('yolo_tf_listener', use_global_arguments=False, cli_args=tf_node_args)
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)

        # YOLO load dinâmico
        self.model = YOLO(self.yolo_model_name)
        self.bridge = CvBridge()

        class_dict = {value: key for key, value in self.model.names.items()}
        self.person_id = class_dict.get('person')

        # Publica a Imagem anotada
        pub_image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.publisher_ = self.create_publisher(Image, f"{ns}/sensors/camera_YOLO", pub_image_qos)
        
        # Publica as Deteccoes (JSON String) para o Gerenciador de Vítimas
        self.victim_pub = self.create_publisher(VictimDetection, f"{ns}/victim_detections", 10)

    def quat_to_mat(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    def callback_camera_info(self, msg: CameraInfo):
        self.camera_info = msg

    def callback_imagens(self, rgb_msg: Image, depth_msg: Image):
        if self.camera_info is None or self.person_id is None:
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            results = self.model.predict(
                source=rgb_image, show=False, conf=0.55, classes=[self.person_id], verbose=False
            )
            result = results[0]

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                yolo_confidence = float(box.conf[0])
                class_name = "person"

                # Centro óptico para a matemática de projeção
                u = int(round((x1 + x2) / 2))
                v = int(round((y1 + y2) / 2))
                u = max(0, min(u, depth_image.shape[1] - 1))
                v = max(0, min(v, depth_image.shape[0] - 1))

                # ========================================================
                # NOVA LÓGICA DE PROFUNDIDADE: Extração Robusta por Região
                # ========================================================
                # 1. Converte limites da Bounding Box para inteiros
                x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
                x2_i, y2_i = min(depth_image.shape[1], int(x2)), min(depth_image.shape[0], int(y2))

                # 2. Recorta toda a matriz de profundidade correspondente à área do YOLO
                depth_roi = depth_image[y1_i:y2_i, x1_i:x2_i].astype(float)

                if depth_msg.encoding == "16UC1":
                    depth_roi /= 1000.0

                # 3. Filtra apenas os pixels que bateram em algo (maior que 10cm e não-infinitos)
                valid_depths = depth_roi[(depth_roi > 0.1) & np.isfinite(depth_roi)]

                if len(valid_depths) == 0:
                    continue

                # 4. Usa o Percentil 30. Ele garante que pegaremos a pessoa (foreground)
                #    e ignoraremos a parede (background) e pequenos ruídos.
                Z = float(np.percentile(valid_depths, 30))
                # ========================================================

                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx = self.camera_info.k[2]
                cy = self.camera_info.k[5]

                # Usa o centro geométrico (u, v) mas com a profundidade (Z) sólida da pessoa
                X_opt = (u - cx) * Z / fx
                Y_opt = (v - cy) * Z / fy
                point_cam = np.array([Z, -X_opt, -Y_opt]) # Z é frente no óptico, X frente no ROS

                try:
                    trans_base = self.tf_buffer.lookup_transform(self.base_frame, "camera_0_link", rclpy.time.Time())
                    trans_global = self.tf_buffer.lookup_transform(self.global_frame, "camera_0_link", rclpy.time.Time())
                except Exception as e:
                    continue

                # Relativo
                t_b = trans_base.transform.translation
                R_b = self.quat_to_mat(trans_base.transform.rotation)
                point_base = R_b.dot(point_cam) + np.array([t_b.x, t_b.y, t_b.z])
                X_base, Y_base, Z_base = point_base
                distancia_relativa_chao = np.sqrt(X_base**2 + Y_base**2)
                angulo_alvo = math.degrees(math.atan2(Y_base, X_base))

                # Global
                t_g = trans_global.transform.translation
                R_g = self.quat_to_mat(trans_global.transform.rotation)
                point_global = R_g.dot(point_cam) + np.array([t_g.x, t_g.y, t_g.z])
                X_glob, Y_glob, Z_glob = point_global

                # Substitua todo o dicionário JSON por uma instância direta e limpa:
                msg = VictimDetection()
                msg.confidence = yolo_confidence
                msg.dist_rel = float(distancia_relativa_chao)
                msg.global_x = float(X_glob)
                msg.global_y = float(Y_glob)
                msg.angle_deg = float(angulo_alvo)

                self.victim_pub.publish(msg)
            
            # Publica Imagem
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
            self.get_logger().error(f"Erro ao processar: {e}\n{traceback.format_exc()}")
            
def main(args=None):
    rclpy.init(args=args)
    node = PrimeiroTesteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'tf_node'): node.tf_node.destroy_node()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()