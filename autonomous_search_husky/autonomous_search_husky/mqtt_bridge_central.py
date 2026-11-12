#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

import paho.mqtt.client as mqtt
import numpy as np
import cv2
import base64
import json
import math

class MqttBridgeCentralNode(Node):
    def __init__(self):
        super().__init__('mqtt_bridge_central_node')
        
        # O IP deve ser o do broker MQTT (provavelmente a própria central)
        self.declare_parameter('broker_ip', '127.0.0.1')
        self.broker_ip = self.get_parameter('broker_ip').value
        self.broker_port = 1883
        
        # Publishers ROS 2 para o RViz
        qos_latch = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        self.map_pub = self.create_publisher(OccupancyGrid, '/central/map', qos_latch)
        self.pose_pub = self.create_publisher(PoseStamped, '/central/robot_pose', 10)
        self.victims_pub = self.create_publisher(MarkerArray, '/central/vitimas_rviz', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Inicialização do MQTT
        self.get_logger().info(f"Conectando ao Broker MQTT em {self.broker_ip}...")
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        self.mqtt_client.connect_async(self.broker_ip, self.broker_port, 10)
        self.mqtt_client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.get_logger().info("✅ Central conectada ao MQTT! Escutando robô...")
            # Assina os tópicos que o robô está enviando
            self.mqtt_client.subscribe("husky/telemetria", qos=0)
            self.mqtt_client.subscribe("husky/mapa", qos=0)
            self.mqtt_client.subscribe("husky/vitimas", qos=1)
        else:
            self.get_logger().error(f"❌ Falha de conexão. Erro: {rc}")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        try:
            if topic == "husky/telemetria":
                self.processar_telemetria(payload)
            elif topic == "husky/mapa":
                self.processar_mapa(payload)
            elif topic == "husky/vitimas":
                self.processar_vitimas(payload)
        except Exception as e:
            self.get_logger().error(f"Erro ao processar {topic}: {e}")

    def processar_telemetria(self, payload):
        dados = json.loads(payload)
        x, y, yaw = dados['x'], dados['y'], dados['yaw']
        agora = self.get_clock().now().to_msg()
        
        # 1. Publica a TF (Transformada Oficial)
        t = TransformStamped()
        t.header.stamp = agora
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        t.transform.rotation.z = float(math.sin(yaw / 2.0))
        t.transform.rotation.w = float(math.cos(yaw / 2.0))
        self.tf_broadcaster.sendTransform(t)
        
        # 2. Publica um PoseStamped (Para desenhar uma Seta no RViz facilmente)
        pose = PoseStamped()
        pose.header.stamp = agora
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation = t.transform.rotation
        self.pose_pub.publish(pose)

    def processar_mapa(self, payload):
        dados = json.loads(payload)
        img_b64 = dados['mapa_png']
        
        # Descompacta Base64 -> Bytes -> OpenCV (Grayscale)
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        if img is None: return
        
        # Reconverte a imagem para o array original do ROS (-1, 0, 100)
        grid = np.full(img.shape, -1, dtype=np.int8)
        grid[img == 255] = 0    # Branco -> Espaço livre (0)
        grid[img == 0] = 100    # Preto -> Obstáculo (100)
        
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.width = dados['width']
        msg.info.height = dados['height']
        msg.info.resolution = dados['res']
        msg.info.origin.position.x = dados['orig_x']
        msg.info.origin.position.y = dados['orig_y']
        msg.data = grid.flatten().tolist()
        
        self.map_pub.publish(msg)

    def processar_vitimas(self, payload):
        dados = json.loads(payload)
        vitimas = dados.get("vitimas", [])
        
        marker_array = MarkerArray()
        
        if not vitimas:
            # Lista vazia = Apagar todos os marcadores da tela
            m = Marker()
            m.header.frame_id = 'map'
            m.action = Marker.DELETEALL
            marker_array.markers.append(m)
        else:
            # Cria uma esfera vermelha para cada vítima
            for v in vitimas:
                m = Marker()
                m.header.frame_id = 'map'
                m.ns = 'vitimas_confirmadas'
                m.id = v['id']
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(v['x'])
                m.pose.position.y = float(v['y'])
                m.pose.position.z = 0.3  # Fica um pouco acima do chão
                m.pose.orientation.w = 1.0
                m.scale.x = 0.6
                m.scale.y = 0.6
                m.scale.z = 0.6
                m.color.r = 1.0
                m.color.a = 1.0 # 100% visível (não transparente)
                marker_array.markers.append(m)
                
        self.victims_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MqttBridgeCentralNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mqtt_client.loop_stop()
        node.mqtt_client.disconnect()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()