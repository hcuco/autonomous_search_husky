import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json

class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager')
        
        # Estado interno para tolerância a falhas
        self.conectado_rede = False
        self.fila_vitimas_offline = []
        
        # --- PUBLISHERS ---
        # Tópico interno: envia as coordenadas para a Bridge despachar via rede
        self.publisher_bridge = self.create_publisher(String, '/husky/enviar_vitima', 10)
        
        # --- SUBSCRIBERS ---
        # 1. Recebe a vítima processada do nó localizador (LiDAR + YOLO)
        self.create_subscription(String, '/husky/vitima_localizada', self.rotear_vitima, 10)
        
        # 2. Escuta ordens da central repassadas pela Bridge
        self.create_subscription(String, '/husky/comando_recebido', self.receber_comando, 10)
        
        # 3. Monitora o status do Wi-Fi/4G (Heartbeat)
        self.create_subscription(Bool, '/status_conexao', self.atualizar_status_rede, 10)
        
        self.get_logger().info("Cérebro da Missão inicializado. Aguardando Bridge MQTT...")

    def atualizar_status_rede(self, msg):
        """Atualiza o estado da conexão e toma decisões de navegação."""
        estava_desconectado = not self.conectado_rede
        self.conectado_rede = msg.data
        
        if self.conectado_rede and estava_desconectado:
            self.get_logger().info("Sinal de rede detectado! Esvaziando fila de dados offline...")
            self.despachar_fila_offline()
            
        elif not self.conectado_rede and not estava_desconectado:
            self.get_logger().warn("Sinal de rede perdido. Modo autônomo offline ativado.")
            # TODO PARA A EQUIPE: 
            # 1. Usar o Action Client do Nav2 para cancelar a meta de exploração atual.
            # 2. Enviar um waypoint de retorno para a base (zona com internet conhecida).

    def receber_comando(self, msg):
        """Recebe ordens de controle da base de operações."""
        self.get_logger().info(f"Comando estratégico recebido: {msg.data}")
        # TODO PARA A EQUIPE:
        # Fazer o parser do JSON (ex: json.loads(msg.data)) e alterar a 
        # Behavior Tree do Nav2 para o algoritmo de busca solicitado.

    def rotear_vitima(self, msg):
        """Avalia o que fazer quando o nó de visão encontra uma vítima."""
        self.get_logger().info("Nova vítima processada pelo sistema.")
        
        if self.conectado_rede:
            # Tem internet: manda direto para o tópico da Bridge
            self.publisher_bridge.publish(msg)
            self.get_logger().info("Vítima repassada para a Bridge MQTT.")
        else:
            # Sem internet: salva na memória RAM (ou banco de dados SQLite)
            self.fila_vitimas_offline.append(msg)
            self.get_logger().warn(f"Coordenada salva offline. Total na fila: {len(self.fila_vitimas_offline)}")

    def despachar_fila_offline(self):
        """Esvazia o armazenamento local enviando tudo quando a rede volta."""
        while self.fila_vitimas_offline:
            msg = self.fila_vitimas_offline.pop(0)
            self.publisher_bridge.publish(msg)
            self.get_logger().info("Dado atrasado publicado na Bridge com sucesso.")

def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Encerrando Gerenciador da Missão...")
    finally:
        node.destroy_node()
        # Prevenção do erro 'rcl_shutdown already called'
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()