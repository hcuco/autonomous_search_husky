import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np

import tf2_ros
from rclpy.qos import qos_profile_sensor_data

class LidarVictimLocalizer(Node):
    def __init__(self):
        super().__init__("lidar_victim_localizer")
        
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")
        
        # Pega automaticamente o ID da classe 'person'
        class_dict = {v: k for k, v in self.model.names.items()}
        self.person_id = class_dict.get('person', 0)

        # Buffer e Listener de TF para descobrir a distância entre o LiDAR e a Câmera
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_info = None
        self.latest_cloud = None

        # Assinaturas (Usando qos_profile_sensor_data que é Best Effort por padrão)
        self.sub_info = self.create_subscription(
            CameraInfo, '/a200_1077/sensors/camera_0/color/camera_info', 
            self.callback_info, qos_profile_sensor_data)
            
        self.sub_cloud = self.create_subscription(
            PointCloud2, '/a200_1077/sensors/lidar3d_0/points', 
            self.callback_cloud, qos_profile_sensor_data)
            
        self.sub_image = self.create_subscription(
            Image, '/a200_1077/sensors/camera_0/color/image', 
            self.callback_image, qos_profile_sensor_data)

        self.get_logger().info("Nó LidarVictimLocalizer iniciado! Aguardando câmera e LiDAR...")

    def callback_info(self, msg: CameraInfo):
        self.camera_info = msg

    def callback_cloud(self, msg: PointCloud2):
        self.latest_cloud = msg

    # Função auxiliar matemática para converter Quaternions (da TF) para Matriz de Rotação
    def quat_to_mat(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    def callback_image(self, rgb_msg: Image):
        if self.camera_info is None or self.latest_cloud is None:
            return

        # 1. Roda o YOLO na imagem recebida
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        results = self.model.predict(source=cv_image, show=False, conf=0.60, classes=[self.person_id], verbose=False)
        result = results[0]

        if len(result.boxes) == 0:
            return

        # 2. Pega a Transformação 3D (TF) do LiDAR para a base física da Câmera
        try:
            # Substituímos o target_frame dinâmico pelo frame que SABEMOS que existe na árvore
            target_frame = "camera_0_link"
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                self.latest_cloud.header.frame_id, # Frame origem: frame do LiDAR
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warning(f"Aguardando TF do LiDAR para a Câmera: {e}")
            return

        # Constrói a Matriz de Transformação 4x4
        t = trans.transform.translation
        R = self.quat_to_mat(trans.transform.rotation)
        T_matrix = np.eye(4)
        T_matrix[:3, :3] = R
        T_matrix[0, 3] = t.x
        T_matrix[1, 3] = t.y
        T_matrix[2, 3] = t.z

        # 3. Lê os pontos do LiDAR e transforma para o ponto de vista da câmera (Padrão ROS)
        points = point_cloud2.read_points_numpy(self.latest_cloud, field_names=("x", "y", "z"), skip_nans=True)
        if points.shape[0] == 0: return

        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack((points, ones))
        points_cam_ros = points_homo.dot(T_matrix.T)[:, :3]

        # -----------------------------------------------------------------
        # PASSO CRÍTICO: Conversão ROS para Óptico (Matemática da Lente)
        # -----------------------------------------------------------------
        # Padrão ROS: X = Frente, Y = Esquerda, Z = Cima
        # Padrão Óptico: Z = Frente, X = Direita, Y = Baixo
        x_opt = -points_cam_ros[:, 1]
        y_opt = -points_cam_ros[:, 2]
        z_opt = points_cam_ros[:, 0]
        
        # Junta os eixos corrigidos de volta num array (N, 3)
        points_cam_opt = np.column_stack((x_opt, y_opt, z_opt))

        # 4. Filtra apenas os pontos que estão à frente da câmera (Z óptico > 0.1m)
        valid_idx = points_cam_opt[:, 2] > 0.1
        pts_valid = points_cam_opt[valid_idx]

        if pts_valid.shape[0] == 0: return

        # 5. Projeta os pontos 3D restantes na imagem (matemática da lente da câmera)
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        cx, cy = self.camera_info.k[2], self.camera_info.k[5]
        
        u = (fx * pts_valid[:, 0] / pts_valid[:, 2]) + cx
        v = (fy * pts_valid[:, 1] / pts_valid[:, 2]) + cy

        # 6. Cruza a projeção com a Bounding Box do YOLO
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            conf = float(box.conf[0])

            # Encontra quais pontos do LiDAR caíram "dentro" do retângulo do YOLO
            in_box_idx = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            pts_in_box = pts_valid[in_box_idx]

            if pts_in_box.shape[0] > 0:
                # Se achou pontos, calcula a mediana para ignorar ruídos (ex: parede atrás da pessoa)
                median_pt = np.median(pts_in_box, axis=0)
                
                # Desfaz a rotação óptica para entregar as coordenadas X, Y, Z no padrão do robô 
                # (onde X é a profundidade real para frente da lente)
                Z_real = median_pt[2]
                Y_real = -median_pt[0]
                X_real = median_pt[2] # Apenas referencial para imprimir

                # A posição do alvo já extraída
                # (Lembrando que na sua lente: Z_real é a profundidade para frente, Y_real é o desvio lateral)
                distancia_horizontal_2d = np.sqrt(Z_real**2 + Y_real**2)

                self.get_logger().info(
                    f"Distância 3D direta (nariz da vítima): {np.linalg.norm([X_real, Y_real, Z_real]):.2f} m\n"
                    f"Distância 2D no chão (pés da vítima): {distancia_horizontal_2d:.2f} m"
                )

                self.get_logger().info(
                    f"\n[VÍTIMA ENCONTRADA (LiDAR Fusion)]\n"
                    f" - Confiança YOLO: {conf:.2f}\n"
                    f" - Pontos do LiDAR atingindo o alvo: {pts_in_box.shape[0]}\n"
                    f" - Distância (Frente): {Z_real:.2f} m\n"
                    f" - Desvio lateral: {Y_real:.2f} m\n"
                    f"----------------------------------------"
                )
            else:
                self.get_logger().warning("Pessoa na imagem, mas o LiDAR não a atingiu.")

def main(args=None):
    rclpy.init(args=args)
    node = LidarVictimLocalizer()
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