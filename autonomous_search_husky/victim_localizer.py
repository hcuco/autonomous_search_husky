import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import random

class VictimLocalizerNode(Node):
    def __init__(self):
        super().__init__('victim_localizer')
        
        # Escuta o gatilho enviado pelo nó do YOLO
        self.create_subscription(String, '/husky/deteccao_visual', self.processar_localizacao, 10)
        
        # Publica a coordenada real para o gerenciador de missão
        self.publisher_ = self.create_publisher(String, '/husky/vitima_localizada', 10)
        
        self.get_logger().info("Nó Localizador de Vítimas inicializado (Aguardando gatilhos).")

    def processar_localizacao(self, msg):
        try:
            dados_visao = json.loads(msg.data)
            self.get_logger().info(f"Gatilho recebido. Calculando distância do pixel ({dados_visao['centro_x']}, {dados_visao['centro_y']})...")

            # TODO PARA A EQUIPE:
            # 1. Assinar os tópicos '/velodyne_points' (LiDAR) e '/navsat/fix' (GPS).
            # 2. Extrair a distância do LiDAR no pixel informado pelo YOLO.
            # 3. Usar o tf2 para somar a distância à posição atual do robô.
            
            # Simulação de coordenadas GPS finais da vítima
            lat_simulada = random.uniform(-10.000, 10.000)
            lon_simulada = random.uniform(-20.000, 20.000)

            dados_finais = {
                "status": "Vítima localizada",
                "latitude": round(lat_simulada, 6),
                "longitude": round(lon_simulada, 6)
            }

            msg_saida = String()
            msg_saida.data = json.dumps(dados_finais)
            
            # Envia a coordenada projetada para o cérebro da missão
            self.publisher_.publish(msg_saida)
            self.get_logger().info("Coordenada calculada e enviada ao Gerenciador.")
            
        except json.JSONDecodeError:
            self.get_logger().error("Erro ao ler dados da visão.")

def main(args=None):
    rclpy.init(args=args)
    node = VictimLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()