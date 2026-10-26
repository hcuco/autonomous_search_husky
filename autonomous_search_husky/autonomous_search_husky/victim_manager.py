import rclpy
from rclpy.node import Node
from autonomous_search_msgs.msg import VictimDetection
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

# exemplo de comando para rodar este nó com parâmetros específicos:
# ros2 run autonomous_search_husky victim_manager --ros-args -p global_frame:=map -p limiar_confirmacao:=10.0 -p limiar_deslocamento_marker:=0.2

class GerenciadorVitimasNode(Node):
    def __init__(self):
        super().__init__("gerenciador_vitimas")

        # Configuração de Parâmetros
        self.declare_parameter("robot_namespace", "/a200_1077")
        self.declare_parameter("raio_agrupamento", 1.5)           # Raio (m) para associar detecções à mesma vítima
        self.declare_parameter("limiar_confirmacao", 3.0)         # Meta de confiança acumulada para confirmar
        self.declare_parameter("limiar_deslocamento_marker", 0.5) # Deslocamento mínimo (m) para mover a esfera no RViz
        self.declare_parameter("global_frame", "odom")            # Frame global ('odom' ou 'map')

        ns = self.get_parameter("robot_namespace").value
        self.raio_agrupamento = self.get_parameter("raio_agrupamento").value
        self.limiar_confirmacao = self.get_parameter("limiar_confirmacao").value
        self.limiar_deslocamento_marker = self.get_parameter("limiar_deslocamento_marker").value
        self.global_frame = self.get_parameter("global_frame").value

        if ns and not ns.startswith('/'): ns = '/' + ns
        if ns.endswith('/'): ns = ns[:-1]

        self.vitimas_potenciais = [] 
        self.vitimas_confirmadas = []
        self.contador_ids = 0

        # Assinatura das detecções customizadas do YOLO
        self.sub = self.create_subscription(
            VictimDetection, f"{ns}/victim_detections", self.callback_deteccao, 10
        )
        
        # Publicador de Marcadores com Durabilidade TRANSIENT_LOCAL (Latching)
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.marker_pub = self.create_publisher(Marker, f"{ns}/victim_markers", marker_qos)

        self.get_logger().info(
            f"Gerenciador de Vítimas ATIVO.\n"
            f" - Limiar de Confirmação: {self.limiar_confirmacao}\n"
            f" - Limiar de Atualização do Marcador: {self.limiar_deslocamento_marker} m"
        )

    def callback_deteccao(self, msg):
        gx = msg.global_x
        gy = msg.global_y
        conf = msg.confidence
        dist_rel = msg.dist_rel
        angulo = getattr(msg, 'angle_deg', 0.0)

        # 1. Checa se a detecção pertence a uma vítima JÁ CONFIRMADA
        for vc in self.vitimas_confirmadas:
            dist = math.hypot(gx - vc['x'], gy - vc['y'])
            if dist <= self.raio_agrupamento:
                # 1.1 Atualiza a posição interna da vítima em tempo real (média móvel)
                vc['atualizacoes'] += 1
                vc['x'] = (vc['x'] * (vc['atualizacoes'] - 1) + gx) / vc['atualizacoes']
                vc['y'] = (vc['y'] * (vc['atualizacoes'] - 1) + gy) / vc['atualizacoes']

                # 1.2 Calcula a distância entre a nova posição interna e a posição visível do marcador no RViz
                deslocamento = math.hypot(vc['x'] - vc['marker_x'], vc['y'] - vc['marker_y'])

                # 1.3 Se o deslocamento for maior que o limiar estipulado, atualiza o marcador no RViz!
                if deslocamento >= self.limiar_deslocamento_marker:
                    vc['marker_x'] = vc['x']
                    vc['marker_y'] = vc['y']
                    self.publicar_marcador(vc['id'], vc['marker_x'], vc['marker_y'])
                    self.get_logger().warn(
                        f" Marcador da Vítima {vc['id']} ATUALIZADO no RViz!\n"
                        f" -> Deslocamento detectado: {deslocamento:.2f} m (>= {self.limiar_deslocamento_marker} m)\n"
                        f" -> Nova Posição no RViz: X = {vc['marker_x']:.2f} | Y = {vc['marker_y']:.2f}"
                    )
                return

        # 2. Checa se pertence a uma vítima POTENCIAL em rastreio
        melhor_vp = None
        menor_dist = float('inf')
        
        for vp in self.vitimas_potenciais:
            dist = math.hypot(gx - vp['x'], gy - vp['y'])
            if dist <= self.raio_agrupamento and dist < menor_dist:
                menor_dist = dist
                melhor_vp = vp

        if melhor_vp is not None:
            # Ponderação de distância e ângulo de ataque
            fator_distancia = max(0.05, 1.0 - (dist_rel / 6.0))
            fator_angulo = max(0.05, 1.0 - (abs(angulo) / 45.0))
            fator_confianca = max(0.05, conf))
            incremento = fator_confianca * fator_distancia * fator_angulo

            melhor_vp['confianca_acumulada'] += incremento
            melhor_vp['atualizacoes'] += 1
            melhor_vp['x'] = (melhor_vp['x'] * (melhor_vp['atualizacoes'] - 1) + gx) / melhor_vp['atualizacoes']
            melhor_vp['y'] = (melhor_vp['y'] * (melhor_vp['atualizacoes'] - 1) + gy) / melhor_vp['atualizacoes']

            if melhor_vp['confianca_acumulada'] >= self.limiar_confirmacao:
                self.confirmar_vitima(melhor_vp, dist_rel)
            else:
                self.get_logger().info(
                    f"Rastreando alvo {melhor_vp['id']}... "
                    f"Acumulado: {melhor_vp['confianca_acumulada']:.2f}/{self.limiar_confirmacao:.2f}"
                )
        else:
            # 3. Registra nova vítima potencial em área inédita
            self.contador_ids += 1
            nova_vp = {
                'id': self.contador_ids,
                'x': gx,
                'y': gy,
                'confianca_acumulada': conf,
                'atualizacoes': 1
            }
            self.vitimas_potenciais.append(nova_vp)
            self.get_logger().info(f"Possível vítima (ID {nova_vp['id']}) avistada. Iniciando acúmulo de confiança...")

    def confirmar_vitima(self, vp, dist_rel):
        # Transfere a vítima para a lista de confirmadas guardando x, y internos e marker_x, marker_y visíveis
        vc = {
            'id': vp['id'],
            'x': vp['x'],
            'y': vp['y'],
            'marker_x': vp['x'],
            'marker_y': vp['y'],
            'atualizacoes': vp['atualizacoes']
        }
        self.vitimas_confirmadas.append(vc)
        self.vitimas_potenciais.remove(vp)
        
        # Publica o marcador inicial no RViz
        self.publicar_marcador(vc['id'], vc['marker_x'], vc['marker_y'])

        self.get_logger().warn(
            f"\n"
            f"VÍTIMA {vc['id']} CONFIRMADA E MARCADA NO RVIZ!\n"
            f" -> Coordenada Global ({self.global_frame}): X = {vc['x']:.2f} | Y = {vc['y']:.2f}\n"
            f"----------------------------------------------------------------"
        )

    def publicar_marcador(self, vitima_id, x, y):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vitimas_confirmadas"
        marker.id = vitima_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 0  # Persiste no RViz
        
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = GerenciadorVitimasNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()