#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import numpy as np
import cv2
import math

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# ros2 run autonomous_search_husky map_coverage --ros-args -p use_sim_time:=true -p alcance_camera_m:=8.0 -p fov_camera_deg:=60.0 -p offset_camera_x:=0.3 -r /tf:=/a200_1077/tf -r /tf_static:=/a200_1077/tf_static


class MapCoverageNode(Node):
    def __init__(self):
        super().__init__('map_coverage_node')

        # Parâmetros da Câmera e Mapeamento
        self.declare_parameter("alcance_camera_m", 7.0)
        self.declare_parameter("fov_camera_deg", 60.0)
        self.declare_parameter("offset_camera_x", 0.3)
        self.declare_parameter("frames_necessarios", 15)
        self.declare_parameter("area_buraco_max_m2", 1.0)

        self.alcance_camera_m = self.get_parameter("alcance_camera_m").value     
        self.fov_camera_rad = math.radians(self.get_parameter("fov_camera_deg").value) 
        self.offset_camera_x = self.get_parameter("offset_camera_x").value
        self.frames_necessarios = self.get_parameter("frames_necessarios").value
        self.area_buraco_max_m2 = self.get_parameter("area_buraco_max_m2").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.mapa_atual = None
        self.memoria_camera = None

        # Subscriber do SLAM
        self.create_subscription(OccupancyGrid, '/a200_1077/map', self.mapa_callback, 10)

        # Publisher da Cobertura Visual
        qos_mapa = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.cobertura_pub = self.create_publisher(OccupancyGrid, '/a200_1077/mapa_cobertura_visual', qos_mapa)
        
        # Timer de Atualização
        self.create_timer(0.2, self.atualizar_memoria_camera)
        self.get_logger().info("✅ Nó de Cobertura Visual iniciado.")

    def mapa_callback(self, msg):
        if self.mapa_atual is None:
            self.get_logger().info(f"✅ MAPA RECEBIDO PARA COBERTURA! Frame: '{msg.header.frame_id}'")

        novo_w = msg.info.width
        novo_h = msg.info.height
        nova_orig = msg.info.origin.position
        res = msg.info.resolution

        # Redimensionamento dinâmico da memória se o mapa do SLAM crescer
        if self.memoria_camera is None:
            self.memoria_camera = np.zeros((novo_h, novo_w), dtype=np.uint8)
        elif self.memoria_camera.shape != (novo_h, novo_w) or \
             self.mapa_atual.info.origin.position.x != nova_orig.x or \
             self.mapa_atual.info.origin.position.y != nova_orig.y:
            
            nova_memoria = np.zeros((novo_h, novo_w), dtype=np.uint8)
            velho_w = self.mapa_atual.info.width
            velho_h = self.mapa_atual.info.height
            velha_orig = self.mapa_atual.info.origin.position
            
            dx = int(round((velha_orig.x - nova_orig.x) / res))
            dy = int(round((velha_orig.y - nova_orig.y) / res))
            
            y_min_n, y_max_n = max(0, dy), min(novo_h, dy + velho_h)
            x_min_n, x_max_n = max(0, dx), min(novo_w, dx + velho_w)
            y_min_v, y_max_v = max(0, -dy), max(0, -dy) + (y_max_n - y_min_n)
            x_min_v, x_max_v = max(0, -dx), max(0, -dx) + (x_max_n - x_min_n)
            
            try:
                nova_memoria[y_min_n:y_max_n, x_min_n:x_max_n] = self.memoria_camera[y_min_v:y_max_v, x_min_v:x_max_v]
            except Exception as e:
                self.get_logger().error(f"Erro ao redimensionar matriz de câmera: {e}")
                
            self.memoria_camera = nova_memoria

        self.mapa_atual = msg

    def get_robot_pose(self):
        if self.mapa_atual is None: return None, None, None
        nome_mapa = self.mapa_atual.header.frame_id
        try:
            t = self.tf_buffer.lookup_transform(nome_mapa, 'a200_1077/base_link', rclpy.time.Time())
        except TransformException:
            try:
                t = self.tf_buffer.lookup_transform(nome_mapa, 'base_link', rclpy.time.Time())
            except TransformException:
                return None, None, None
        
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return x, y, yaw

    def atualizar_memoria_camera(self):
        if self.mapa_atual is None or self.memoria_camera is None:
            return

        if not hasattr(self, 'contador_exposicao') or self.contador_exposicao.shape != self.memoria_camera.shape:
            self.contador_exposicao = np.zeros_like(self.memoria_camera, dtype=np.int16)

        rx, ry, ryaw = self.get_robot_pose()
        if rx is not None:
            res = self.mapa_atual.info.resolution
            origem_x = self.mapa_atual.info.origin.position.x
            origem_y = self.mapa_atual.info.origin.position.y
            w, h = self.mapa_atual.info.width, self.mapa_atual.info.height
            grid_slam = np.array(self.mapa_atual.data).reshape((h, w))

            cam_x = rx + self.offset_camera_x * math.cos(ryaw)
            cam_y = ry + self.offset_camera_x * math.sin(ryaw)
            cx_px, cy_px = (cam_x - origem_x) / res, (cam_y - origem_y) / res

            raio_px = int(self.alcance_camera_m / res)
            num_raios = 150 
            ang_inicial = ryaw - self.fov_camera_rad / 2.0
            passo_ang = self.fov_camera_rad / num_raios

            visto_neste_frame = np.zeros_like(self.memoria_camera, dtype=bool)

            for i in range(num_raios):
                ang = ang_inicial + i * passo_ang
                dx, dy = math.cos(ang), math.sin(ang)
                for r in range(raio_px):
                    px, py = int(cx_px + r * dx), int(cy_px + r * dy)
                    if 0 <= px < w and 0 <= py < h:
                        if grid_slam[py, px] > 50: break
                        visto_neste_frame[py, px] = True
                    else: break

            self.contador_exposicao[visto_neste_frame] += 1
            pixels_confirmados = self.contador_exposicao >= self.frames_necessarios
            self.memoria_camera[pixels_confirmados] = 1

            raio_robo_pixels = int(0.7 / res)
            centro_robo = (int((rx - origem_x) / res), int((ry - origem_y) / res))
            cv2.circle(self.memoria_camera, centro_robo, raio_robo_pixels, 1, -1)

            # Preenchimento de pequenos buracos
            area_maxima_px = int(self.area_buraco_max_m2 / (res * res))
            mask_nao_explorado = np.uint8(self.memoria_camera == 0)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_nao_explorado, connectivity=8)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] <= area_maxima_px:
                    self.memoria_camera[labels == i] = 1
                    self.contador_exposicao[labels == i] = self.frames_necessarios

        # Publica o resultado
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.mapa_atual.header.frame_id
        msg.info = self.mapa_atual.info
        
        grid_cobertura = np.full(self.memoria_camera.shape, -1, dtype=np.int8)
        grid_cobertura[self.memoria_camera == 1] = 0 
        msg.data = grid_cobertura.flatten().tolist()
        self.cobertura_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapCoverageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Encerrando nó de cobertura de mapa...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()