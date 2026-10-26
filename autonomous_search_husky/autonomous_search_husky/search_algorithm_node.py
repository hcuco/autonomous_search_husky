import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Bool
import numpy as np
import json
import cv2
import threading
import time

class SearchAlgorithmNode(Node):
    def __init__(self):
        super().__init__('search_algorithm_node')

        # 1. Inicializa o navegador oficial do Nav2
        self.navigator = BasicNavigator(namespace='a200_1077')
        self.get_logger().info("Aguardando ativação do Nav2 e SLAM...")
        self.navigator.waitUntilNav2Active(localizer='slam_toolbox')

        # 2. Estado interno da missão
        self.algoritmo_atual = "busca_fronteira"
        self.conectado_rede = True
        self.em_missao = False
        self.mapa_atual = None
        self.mudou_algoritmo = False
        self.voltar_base = False
        self.base_segura_pose = self.criar_pose(0.0, 0.0)

        # 3. Subscribers (Apenas recebem dados, NÃO bloqueiam o código)
        self.create_subscription(OccupancyGrid, '/a200_1077/map', self.mapa_callback, 10)
        self.create_subscription(String, '/husky/comando_recebido', self.receber_comando, 10)
        self.create_subscription(Bool, '/status_conexao', self.atualizar_status_rede, 10)

        self.get_logger().info("Estratega de Busca inicializado com sucesso.")

    def criar_pose(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        return pose

    # ==========================================
    # CALLBACKS (Executados em Background)
    # ==========================================
    def mapa_callback(self, msg):
        self.mapa_atual = msg

    def atualizar_status_rede(self, msg):
        estava_conectado = self.conectado_rede
        self.conectado_rede = msg.data
        # Se a rede cair, sinaliza para a Thread principal tomar atitude
        if not self.conectado_rede and estava_conectado:
            self.voltar_base = True

    def receber_comando(self, msg):
        try:
            comando = json.loads(msg.data)
            if comando.get("acao") == "iniciar_busca":
                self.algoritmo_atual = comando.get("algoritmo", "busca_fronteira")
                self.get_logger().info(f"Ordem recebida. Algoritmo alterado para: {self.algoritmo_atual}")
                self.mudou_algoritmo = True
        except json.JSONDecodeError:
            pass

    # ==========================================
    # MATEMÁTICA DE BUSCA
    # ==========================================
    def calcular_fronteira(self):
        if self.mapa_atual is None:
            return None
        
        w = self.mapa_atual.info.width
        h = self.mapa_atual.info.height
        res = self.mapa_atual.info.resolution
        origem_x = self.mapa_atual.info.origin.position.x
        origem_y = self.mapa_atual.info.origin.position.y

        grid = np.array(self.mapa_atual.data).reshape((h, w))
        espaco_livre = np.uint8(grid == 0)
        espaco_desconhecido = np.uint8(grid == -1)

        kernel = np.ones((3, 3), np.uint8)
        livre_expandido = cv2.dilate(espaco_livre, kernel, iterations=1)
        matriz_fronteiras = cv2.bitwise_and(livre_expandido, espaco_desconhecido)

        pontos_y, pontos_x = np.where(matriz_fronteiras > 0)

        if len(pontos_x) == 0:
            return None

        idx_escolhido = np.random.randint(0, len(pontos_x))
        alvo_x_metros = (pontos_x[idx_escolhido] * res) + origem_x
        alvo_y_metros = (pontos_y[idx_escolhido] * res) + origem_y

        return self.criar_pose(alvo_x_metros, alvo_y_metros)

    def calcular_espiral(self):
        return self.criar_pose(1.5, -1.5)

    def calcular_aleatorio(self):
        x_random = float(np.random.uniform(0.0, 3.0))
        y_random = float(np.random.uniform(-3.0, 3.0))
        return self.criar_pose(x_random, y_random)


# ==========================================
# LOOP PRINCIPAL (Protegido de Travamentos)
# ==========================================
def main(args=None):
    rclpy.init(args=args)
    node = SearchAlgorithmNode()

    # 1. A GRANDE MUDANÇA: Cria um "motor" exclusivo para este nó
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # 2. Isola os Callbacks na Thread usando o motor exclusivo (e não o global)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # 3. O Roteador Principal (Onde o Nav2 trabalha livremente)
        while rclpy.ok():
            
            # A) Protocolo de Emergência: Rede Caiu
            if node.voltar_base:
                node.get_logger().warn("Sinal perdido! Cancelando busca e voltando à base...")
                node.navigator.cancelTask()
                node.navigator.goToPose(node.base_segura_pose)
                node.em_missao = True
                node.voltar_base = False
            
            # B) Nova ordem da Central: Interromper rota atual
            elif node.mudou_algoritmo and node.conectado_rede:
                node.get_logger().info("Cancelando rota atual para aplicar novo padrão geométrico...")
                node.navigator.cancelTask()
                node.em_missao = False
                node.mudou_algoritmo = False

            # C) Ação Padrão: Despachar nova coordenada caso esteja livre
            elif not node.em_missao and node.conectado_rede and node.mapa_atual is not None:
                if node.algoritmo_atual == "busca_fronteira":
                    alvo = node.calcular_fronteira()
                elif node.algoritmo_atual == "busca_espiral":
                    alvo = node.calcular_espiral()
                else:
                    alvo = node.calcular_aleatorio()

                if alvo:
                    node.get_logger().info(f"Despachando Nav2 para: X={alvo.pose.position.x:.2f}, Y={alvo.pose.position.y:.2f}")
                    node.navigator.goToPose(alvo)
                    node.em_missao = True

            # D) Aguardar: O robô já terminou a viagem?
            if node.em_missao:
                if node.navigator.isTaskComplete():
                    node.get_logger().info("Destino alcançado! Extraindo nova matriz...")
                    node.em_missao = False

            time.sleep(0.5)

    except KeyboardInterrupt:
        node.get_logger().info("Encerrando algoritmo de busca...")
    finally:
        node.navigator.cancelTask()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()