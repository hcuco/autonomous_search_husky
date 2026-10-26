import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class LidarTestNode(Node):
    def __init__(self):
        super().__init__('lidar_tester')
        
        # O tópico exato do LiDAR Ouster na sua simulação
        self.lidar_topic = '/a200_1077/sensors/lidar3d_0/points' #
        
        self.get_logger().info(f"Iniciando nó de teste. Aguardando dados em: {self.lidar_topic}")
        
        # Inscrição (Subscriber) para receber a nuvem de pontos
        self.subscription = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg):
        """Função disparada toda vez que o LiDAR emite uma nova leitura do ambiente."""
        # Apenas imprime um resumo dos dados para não floodar o terminal
        largura = msg.width
        altura = msg.height
        
        self.get_logger().info(f"Nuvem de pontos recebida! Resolução: {largura} x {altura} pontos.")

def main(args=None):
    rclpy.init(args=args)
    node = LidarTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Encerrando teste do LiDAR...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()