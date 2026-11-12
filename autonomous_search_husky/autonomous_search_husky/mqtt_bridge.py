#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import paho.mqtt.client as mqtt
import numpy as np
import cv2
import base64
import json
import math

# Lembre-se de rodar com o remap de TF para a telemetria funcionar:
# ros2 run autonomous_search_husky mqtt_bridge --ros-args -p use_sim_time:=true -p broker_ip:=127.0.0.1 -r /tf:=/a200_1077/tf -r /tf_static:=/a200_1077/tf_static


class MqttBridgeNode(Node):
    def __init__(self):
        super().__init__('mqtt_bridge_node')
        
        # Estado interno de conexão da ponte e cache de mapa
        self.is_connected = False
        self.mapa_atual = None
        
        # 1. Parâmetros de Conexão MQTT
        self.declare_parameter('broker_ip', '127.0.0.1')
        self.broker_ip = self.get_parameter('broker_ip').value
        self.broker_port = 1883
        
        # Tópicos externos (MQTT)
        self.mqtt_topic_vitimas = "husky/vitimas"
        self.mqtt_topic_telemetria = "husky/telemetria"
        self.mqtt_topic_mapa = "husky/mapa"
        self.mqtt_topic_sub = "central/comandos"  
        
        # 2. Configuração de TF2 (para leitura da pose do robô)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 3. Configuração dos Tópicos Internos (ROS 2)
        self.ros_subscriber = self.create_subscription(
            String, 
            '/husky/enviar_vitima', 
            self.ros_to_mqtt_callback, 
            10
        )
        
        qos_mapa = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.mapa_subscriber = self.create_subscription(
            OccupancyGrid,
            '/a200_1077/map',
            self.mapa_callback,
            qos_mapa
        )
        
        self.ros_publisher = self.create_publisher(
            String, 
            '/husky/comando_recebido', 
            10
        )
        
        # Timers periódicos (Heartbeat, Telemetria e Mapa)
        self.status_publisher = self.create_publisher(Bool, '/status_conexao', 10)
        self.status_timer = self.create_timer(1.0, self.publicar_status_conexao)
        self.telemetria_timer = self.create_timer(0.25, self.enviar_telemetria_mqtt)  # 4 Hz
        self.mapa_timer = self.create_timer(2.5, self.enviar_mapa_mqtt)              # 0.4 Hz
        
        # 4. Inicialização Resiliente do Cliente MQTT
        self.get_logger().info(f"Inicializando Bridge MQTT. Apontando para Broker: {self.broker_ip}")
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        self.mqtt_client.will_set("husky/status", '{"online": false}', qos=1, retain=True)
        
        self.mqtt_client.connect_async(self.broker_ip, self.broker_port, keepalive=10)
        self.mqtt_client.loop_start()

    # ==========================================
    # FLUXO 1: MONITORAMENTO DE REDE (HEARTBEAT)
    # ==========================================
    def publicar_status_conexao(self):
        msg = Bool()
        msg.data = self.is_connected
        self.status_publisher.publish(msg)

    # ==========================================
    # FLUXO 2: RECEBENDO DO MQTT -> ENVIANDO PARA ROS 2
    # ==========================================
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            self.get_logger().info("Bridge conectada ao Broker da Central! Escutando comandos...")
            self.mqtt_client.publish("husky/status", '{"online": true}', qos=1, retain=True)
            self.mqtt_client.subscribe(self.mqtt_topic_sub, qos=1)
        else:
            self.is_connected = False
            self.get_logger().error(f"Falha na conexão MQTT. Código: {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        self.get_logger().warn("Conexão MQTT perdida com a central.")

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode('utf-8')
        self.get_logger().info(f"[MQTT -> ROS 2] Ordem recebida: {payload}")
        
        msg_ros = String()
        msg_ros.data = payload
        self.ros_publisher.publish(msg_ros)

    # ==========================================
    # FLUXO 3: VÍTIMAS (ROS 2 -> MQTT)
    # ==========================================
    def ros_to_mqtt_callback(self, msg):
        if self.is_connected:
            self.mqtt_client.publish(self.mqtt_topic_vitimas, msg.data, qos=1)

    # ==========================================
    # FLUXO 4: TELEMETRIA DO ROBÔ (TF2 -> MQTT)
    # ==========================================
    def get_robot_pose(self):
        nome_mapa = self.mapa_atual.header.frame_id if self.mapa_atual else 'a200_1077/map'
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
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return x, y, yaw

    def enviar_telemetria_mqtt(self):
        if not self.is_connected:
            return
            
        x, y, yaw = self.get_robot_pose()
        if x is None:
            return
            
        payload = {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "yaw": round(float(yaw), 4)
        }
        # qos=0 evita acúmulo de fila caso o Wi-Fi oscile
        self.mqtt_client.publish(self.mqtt_topic_telemetria, json.dumps(payload), qos=0)

    # ==========================================
    # FLUXO 5: COMPACTAÇÃO E ENVIO DO MAPA
    # ==========================================
    def mapa_callback(self, msg):
        self.mapa_atual = msg

    def enviar_mapa_mqtt(self):
        if not self.is_connected or self.mapa_atual is None:
            return
            
        w = self.mapa_atual.info.width
        h = self.mapa_atual.info.height
        if w == 0 or h == 0:
            return
            
        # Converte OccupancyGrid (-1, 0, 100) para imagem em escala de cinza (127, 255, 0)
        grid = np.array(self.mapa_atual.data, dtype=np.int8).reshape((h, w))
        img = np.full((h, w), 127, dtype=np.uint8)  # Desconhecido (-1) -> Cinza
        img[grid == 0] = 255                        # Livre (0) -> Branco
        img[grid >= 50] = 0                         # Obstáculo (>=50) -> Preto
        
        # Compacta em PNG na memória e codifica em Base64
        sucesso, buffer = cv2.imencode('.png', img)
        if not sucesso:
            self.get_logger().error("Falha ao compactar mapa em PNG.")
            return
            
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "frame_id": self.mapa_atual.header.frame_id,
            "width": int(w),
            "height": int(h),
            "res": round(float(self.mapa_atual.info.resolution), 4),
            "orig_x": round(float(self.mapa_atual.info.origin.position.x), 4),
            "orig_y": round(float(self.mapa_atual.info.origin.position.y), 4),
            "mapa_png": img_b64
        }
        
        self.mqtt_client.publish(self.mqtt_topic_mapa, json.dumps(payload), qos=0)


def main(args=None):
    rclpy.init(args=args)
    node = MqttBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Encerrando Bridge MQTT...")
    finally:
        node.mqtt_client.loop_stop()
        node.mqtt_client.disconnect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()