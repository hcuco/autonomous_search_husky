import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import paho.mqtt.client as mqtt

class MqttBridgeNode(Node):
    def __init__(self):
        super().__init__('mqtt_bridge_node')
        
        # Estado interno de conexão da ponte
        self.is_connected = False
        
        # 1. Parâmetros de Conexão MQTT
        self.declare_parameter('broker_ip', '127.0.0.1')
        self.broker_ip = self.get_parameter('broker_ip').value
        self.broker_port = 1883
        
        # Tópicos externos (MQTT)
        self.mqtt_topic_pub = "husky/vitimas"     
        self.mqtt_topic_sub = "central/comandos"  
        
        # 2. Configuração dos Tópicos Internos (ROS 2)
        self.ros_subscriber = self.create_subscription(
            String, 
            '/husky/enviar_vitima', 
            self.ros_to_mqtt_callback, 
            10
        )
        
        self.ros_publisher = self.create_publisher(
            String, 
            '/husky/comando_recebido', 
            10
        )
        
        # Publisher e Timer para o Heartbeat de rede
        self.status_publisher = self.create_publisher(Bool, '/status_conexao', 10)
        self.status_timer = self.create_timer(1.0, self.publicar_status_conexao)
        
        # 3. Inicialização do Cliente MQTT
        self.get_logger().info(f"Inicializando Bridge MQTT. Apontando para Broker: {self.broker_ip}")
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        try:
            self.mqtt_client.connect(self.broker_ip, self.broker_port, keepalive=10)
            self.mqtt_client.loop_start()
        except Exception as e:
            self.get_logger().error(f"Erro de rota ao conectar no MQTT: {e}")

    # ==========================================
    # FLUXO 1: MONITORAMENTO DE REDE (HEARTBEAT)
    # ==========================================
    def publicar_status_conexao(self):
        """Publica repetidamente True ou False dependendo do estado do MQTT."""
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
            self.mqtt_client.subscribe(self.mqtt_topic_sub)
        else:
            self.is_connected = False
            self.get_logger().error(f"Falha na conexão MQTT. Código: {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        self.get_logger().warn("Conexão MQTT perdida com a central.")

    def on_message(self, client, userdata, msg):
        """Dispara quando a central manda uma mensagem MQTT. Repassa para o ROS 2."""
        payload = msg.payload.decode('utf-8')
        self.get_logger().info(f"[MQTT -> ROS 2] Ordem recebida: {payload}")
        
        msg_ros = String()
        msg_ros.data = payload
        self.ros_publisher.publish(msg_ros)

    # ==========================================
    # FLUXO 3: RECEBENDO DO ROS 2 -> ENVIANDO PARA MQTT
    # ==========================================
    def ros_to_mqtt_callback(self, msg):
        """Dispara quando o nó de visão/gerenciador envia dados da vítima. Repassa para MQTT."""
        if self.is_connected:
            self.get_logger().info(f"[ROS 2 -> MQTT] Mapeando vítima para a central: {msg.data}")
            self.mqtt_client.publish(self.mqtt_topic_pub, msg.data, qos=1)
        else:
            self.get_logger().warn("Tentativa de envio falhou: Sem rede disponível.")


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