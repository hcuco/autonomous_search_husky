import rclpy
from rclpy.node import Node
import paho.mqtt.client as mqtt
import json

class HuskyMqttNode(Node):
    def __init__(self):
        super().__init__('husky_mqtt_tester')

        self.declare_parameter("broker_ip", "150.162.184.178")
        
        # 1. Configurações do MQTT
        # self.broker_ip = "192.168.134.164"  # IP da sua central
        self.broker_ip = self.get_parameter("broker_ip").value
        self.broker_port = 1883
        self.mqtt_topic = "husky/vitimas"
        
        self.get_logger().info(f"Inicializando Cliente MQTT no IP: {self.broker_ip}...")
        self.mqtt_client = mqtt.Client()
        
        # Conectando callbacks para monitorar a conexão
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Conecta ao Broker
        try:
            self.mqtt_client.connect(self.broker_ip, self.broker_port, keepalive=60)
            self.mqtt_client.loop_start()  # Roda a rede do MQTT em background
        except Exception as e:
            self.get_logger().error(f"Erro ao conectar no MQTT: {e}")

        # 2. Timer do ROS2 (Simulando a detecção de vítimas a cada 5 segundos)
        self.timer = self.create_timer(5.0, self.simular_deteccao)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.get_logger().info(f"Conectado com sucesso ao Broker na Central ({self.broker_ip})!")
        else:
            self.get_logger().error(f"Falha na conexão MQTT. Código de erro: {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.get_logger().warn("Conexão com a Central perdida!")

    def simular_deteccao(self):
        # Criando um payload estruturado (fácil de ler na central)
        dados_vitima = {
            "status": "Vítima detectada (Simulação)",
            "latitude": -27.6012,
            "longitude": -48.5520,
            "bateria_husky": "78%"
        }
        
        # Converte o dicionário Python para string JSON
        mensagem = json.dumps(dados_vitima)
        
        # Publica no MQTT
        self.mqtt_client.publish(self.mqtt_topic, mensagem)
        self.get_logger().info(f"Enviado para a central: {mensagem}")

def main(args=None):
    rclpy.init(args=args)
    node = HuskyMqttNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Encerrando o nó de teste MQTT...")
    finally:
        # Desliga o MQTT de forma limpa ao fechar o nó
        node.mqtt_client.loop_stop()
        node.mqtt_client.disconnect()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()