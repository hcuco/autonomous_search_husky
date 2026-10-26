import rclpy
from rclpy.node import Node
import paho.mqtt.client as mqtt
import socket
import json

class MqttCentralSubscriber(Node):
    def __init__(self):
        super().__init__('mqtt_central_server')
        
        # 1. Descobre e printa o IP da Central no terminal
        self.ip_central = self.obter_ip_local()
        self.get_logger().info(f"=== SERVIDOR MQTT INICIADO ===")
        self.get_logger().info(f"IP da Central (Broker): {self.ip_central}")
        
        # 2. Configurações do MQTT
        self.broker_ip = "localhost" # Como roda na própria central, conecta a si mesmo
        self.broker_port = 1883
        self.mqtt_topic = "husky/vitimas"
        
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # 3. Conecta ao Broker e inicia a thread de escuta
        try:
            self.mqtt_client.connect(self.broker_ip, self.broker_port, keepalive=60)
            self.mqtt_client.loop_start()
        except Exception as e:
            self.get_logger().error(f"Erro ao conectar ao Broker local: {e}")

    def obter_ip_local(self):
        """Abre uma conexão temporária para descobrir o IP real da máquina na rede Wi-Fi/Roteador"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Não precisa alcançar a internet, apenas simula uma rota
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.get_logger().info(f"Ouvindo mensagens no tópico: '{self.mqtt_topic}'...")
            self.mqtt_client.subscribe(self.mqtt_topic)
        else:
            self.get_logger().error("Falha na conexão com o Broker.")

    def on_message(self, client, userdata, msg):
        # Esta função dispara automaticamente sempre que o Husky envia uma vítima
        payload = msg.payload.decode('utf-8')
        
        try:
            # Formata o JSON para ficar fácil de ler no terminal
            dados = json.loads(payload)
            self.get_logger().info(f"\n[ALERTA] Vítima recebida do Husky:\n"
                                   f"Status: {dados.get('status')}\n"
                                   f"Lat: {dados.get('latitude')} | Lon: {dados.get('longitude')}")
        except json.JSONDecodeError:
            self.get_logger().info(f"Mensagem bruta recebida: {payload}")

def main(args=None):
    rclpy.init(args=args)
    node = MqttCentralSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Encerrando a escuta da central...")
    finally:
        node.mqtt_client.loop_stop()
        node.mqtt_client.disconnect()
        node.destroy_node()
        # Verificação para evitar o bug de fechamento duplicado do ROS2
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()