import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Tópico onde este nó avisa que encontrou uma pessoa na imagem
        self.publisher_ = self.create_publisher(String, '/husky/deteccao_visual', 10)
        
        # Simulação: Dispara a detecção a cada 5 segundos
        self.timer = self.create_timer(5.0, self.simular_deteccao)
        
        self.get_logger().info("Nó YOLO inicializado (Modo Simulação). Aguardando detecções...")

    def simular_deteccao(self):
        """Simula a saída da rede neural."""
        # Payload simulando o centro da caixa delimitadora (bounding box) e a classe
        dados_visao = '{"alvo": "vitima", "confianca": 0.88, "centro_x": 320, "centro_y": 240}'
        
        msg = String()
        msg.data = dados_visao
        self.publisher_.publish(msg)
        
        self.get_logger().info(f"Simulando pessoa detectada: {dados_visao}")
        
        # TODO PARA A EQUIPE: 
        # 1. Remover este timer.
        # 2. Criar um Subscriber para o tópico '/camera/image_raw'.
        # 3. Rodar 'resultados = self.modelo(frame)'.
        # 4. Publicar a mensagem apenas quando a classe for 'vitima'.

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
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