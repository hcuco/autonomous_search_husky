import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Bool
import numpy as np
import json
import cv2
import math
import threading
import time

# ros2 run autonomous_search_husky search_algorithm_node_camera --ros-args -p use_sim_time:=true -r /tf:=/a200_1077/tf -r /tf_static:=/a200_1077/tf_static
# ros2 run autonomous_search_husky search_algorithm_node_camera --ros-args -p use_sim_time:=true -p timeout_missao_segundos:=20.0 -p alcance_camera_m:=6.0 -p fov_camera_deg:=60.0 -p offset_camera_x:=0.3 -r /tf:=/a200_1077/tf -r /tf_static:=/a200_1077/tf_static


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class SearchAlgorithmNode(Node):
    def __init__(self, navigator):
        super().__init__('search_algorithm_node')

        # Configuração de Parâmetros
        self.declare_parameter("timeout_missao_segundos", 20.0)
        self.declare_parameter("alcance_camera_m", 7.0)           # Raio (m) para associar detecções à mesma vítima
        self.declare_parameter("fov_camera_deg", 60.0)         # Meta de confiança acumulada para confirmar
        self.declare_parameter("offset_camera_x", 0.3) # Deslocamento mínimo (m) para mover a esfera no RViz
        self.declare_parameter("limite_distancia_nav", 4.0)

        self.navigator = navigator
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Estado interno
        self.algoritmo_atual = "busca_fronteira"
        self.conectado_rede = True
        self.em_missao = False
        self.mapa_atual = None
        self.mudou_algoritmo = False
        self.voltar_base = False
        self.base_segura_pose = self.criar_pose(0.0, 0.0)
        
        # SISTEMA ANTI-TRAVAMENTO
        self.tempo_inicio_missao = None
        self.timeout_missao_segundos = self.get_parameter("timeout_missao_segundos").value   
        self.ultimo_alvo_px = None 

        # MATRIZ QUE GUARDA A COBERTURA VISUAL
        self.memoria_camera = None

        # --- NOVO: LIMITADOR DE DISTÂNCIA (WAYPOINT INTERMEDIÁRIO) ---
        # Raio máximo seguro baseado no tamanho padrão de um local costmap do Nav2 (ex: 2.0 a 2.5 metros)
        self.limite_distancia_nav = self.get_parameter("limite_distancia_nav").value

        # ==========================================
        # CONFIGURAÇÕES FÍSICAS DA CÂMERA
        # ==========================================
        self.alcance_camera_m = self.get_parameter("alcance_camera_m").value     
        self.fov_camera_rad = math.radians(self.get_parameter("fov_camera_deg").value) 
        self.offset_camera_x = self.get_parameter("offset_camera_x").value    

        # Subscribers
        self.create_subscription(OccupancyGrid, '/a200_1077/map', self.mapa_callback, 10)
        self.create_subscription(String, '/husky/comando_recebido', self.receber_comando, 10)
        self.create_subscription(Bool, '/status_conexao', self.atualizar_status_rede, 10)

        # Publicador
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        qos_mapa = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.cobertura_pub = self.create_publisher(OccupancyGrid, '/a200_1077/mapa_cobertura_visual', qos_mapa)
        
        self.create_timer(0.2, self.atualizar_memoria_camera)

    def criar_pose(self, x, y, yaw=0.0):
        pose = PoseStamped()
        nome_mapa = self.mapa_atual.header.frame_id if self.mapa_atual else 'a200_1077/map'
        pose.header.frame_id = nome_mapa
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        
        pose.pose.orientation.z = float(math.sin(yaw / 2.0))
        pose.pose.orientation.w = float(math.cos(yaw / 2.0))
        return pose

    # ==========================================
    # GERENCIAMENTO DO MAPA
    # ==========================================
    def mapa_callback(self, msg):
        if self.mapa_atual is None:
            self.get_logger().info(f"✅ MAPA RECEBIDO! Frame: '{msg.header.frame_id}'")

        novo_w = msg.info.width
        novo_h = msg.info.height
        nova_orig = msg.info.origin.position
        res = msg.info.resolution

        if self.memoria_camera is None:
            self.memoria_camera = np.zeros((novo_h, novo_w), dtype=np.uint8)
        elif self.memoria_camera.shape != (novo_h, novo_w) or \
             self.mapa_atual.info.origin.position.x != nova_orig.x or \
             self.mapa_atual.info.origin.position.y != nova_orig.y:
            
            nova_memoria = np.zeros((novo_h, novo_w), dtype=np.uint8)
            velho_w = self.mapa_atual.info.width
            velho_h = self.mapa_atual.info.height
            velha_orig = self.mapa_atual.info.origin.position
            
            dx = int(round((velha_orig.x - nova_orig.x) / res))
            dy = int(round((velha_orig.y - nova_orig.y) / res))
            
            y_min_n = max(0, dy)
            y_max_n = min(novo_h, dy + velho_h)
            x_min_n = max(0, dx)
            x_max_n = min(novo_w, dx + velho_w)
            
            y_min_v = max(0, -dy)
            y_max_v = y_min_v + (y_max_n - y_min_n)
            x_min_v = max(0, -dx)
            x_max_v = x_min_v + (x_max_n - x_min_n)
            
            try:
                nova_memoria[y_min_n:y_max_n, x_min_n:x_max_n] = \
                    self.memoria_camera[y_min_v:y_max_v, x_min_v:x_max_v]
            except Exception as e:
                self.get_logger().error(f"Erro ao redimensionar matriz: {e}")
                
            self.memoria_camera = nova_memoria

        self.mapa_atual = msg

    def atualizar_status_rede(self, msg):
        estava_conectado = self.conectado_rede
        self.conectado_rede = msg.data
        if not self.conectado_rede and estava_conectado:
            self.voltar_base = True

    def receber_comando(self, msg):
        try:
            comando = json.loads(msg.data)
            if comando.get("acao") == "iniciar_busca":
                self.algoritmo_atual = comando.get("algoritmo", "busca_fronteira")
                self.mudou_algoritmo = True
        except json.JSONDecodeError:
            pass

    def get_robot_pose(self):
        if self.mapa_atual is None:
            return None, None, None
            
        nome_mapa = self.mapa_atual.header.frame_id
        try:
            t = self.tf_buffer.lookup_transform(nome_mapa, 'a200_1077/base_link', rclpy.time.Time())
        except TransformException:
            try:
                t = self.tf_buffer.lookup_transform(nome_mapa, 'base_link', rclpy.time.Time())
            except TransformException as e:
                self.get_logger().warn(f"⚠️ ERRO DE TF! ({e})", throttle_duration_sec=3.0)
                return None, None, None
                
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return x, y, yaw

    # ==========================================
    # CÂMERA INTELIGENTE (Raycasting para não ver através da parede)
    # ==========================================
    def atualizar_memoria_camera(self):
        if self.mapa_atual is None or self.memoria_camera is None:
            return

        rx, ry, ryaw = self.get_robot_pose()

        if rx is not None:
            res = self.mapa_atual.info.resolution
            origem_x = self.mapa_atual.info.origin.position.x
            origem_y = self.mapa_atual.info.origin.position.y
            w = self.mapa_atual.info.width
            h = self.mapa_atual.info.height
            grid_slam = np.array(self.mapa_atual.data).reshape((h, w))

            cam_x = rx + self.offset_camera_x * math.cos(ryaw)
            cam_y = ry + self.offset_camera_x * math.sin(ryaw)
            cx_px = (cam_x - origem_x) / res
            cy_px = (cam_y - origem_y) / res

            raio_px = int(self.alcance_camera_m / res)
            num_raios = 150 
            ang_inicial = ryaw - self.fov_camera_rad / 2.0
            passo_ang = self.fov_camera_rad / num_raios

            for i in range(num_raios):
                ang = ang_inicial + i * passo_ang
                dx = math.cos(ang)
                dy = math.sin(ang)
                
                for r in range(raio_px):
                    px = int(cx_px + r * dx)
                    py = int(cy_px + r * dy)
                    
                    if 0 <= px < w and 0 <= py < h:
                        if grid_slam[py, px] > 50: 
                            break
                        self.memoria_camera[py, px] = 1
                    else:
                        break

            # Sombra para o próprio corpo do robô
            raio_robo_pixels = int(0.7 / res)
            centro_robo = (int((rx - origem_x) / res), int((ry - origem_y) / res))
            cv2.circle(self.memoria_camera, centro_robo, raio_robo_pixels, 1, -1)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.mapa_atual.header.frame_id
        msg.info = self.mapa_atual.info
        grid_cobertura = np.full(self.memoria_camera.shape, -1, dtype=np.int8)
        grid_cobertura[self.memoria_camera == 1] = 0 
        msg.data = grid_cobertura.flatten().tolist()
        self.cobertura_pub.publish(msg)

    # ==========================================
    # LÓGICA DE EXPLORAÇÃO HÍBRIDA (Gira Perto + Proteção contra Parede)
    # ==========================================
    def calcular_fronteira(self):
        if self.mapa_atual is None or self.memoria_camera is None:
            return None

        w = self.mapa_atual.info.width
        h = self.mapa_atual.info.height
        res = self.mapa_atual.info.resolution
        origem_x = self.mapa_atual.info.origin.position.x
        origem_y = self.mapa_atual.info.origin.position.y

        grid = np.array(self.mapa_atual.data).reshape((h, w))
        
        # 1. SEGURANÇA DINÂMICA (Husky tem ~67cm de largura, pedimos 55cm de raio de segurança)
        raio_seguranca_m = 0.55
        raio_px = max(1, int(raio_seguranca_m / res))
        tam_kernel = raio_px * 2 + 1
        
        espaco_livre = np.uint8(grid == 0) * 255
        obstaculos = np.uint8(grid >= 50) * 255
        
        kernel_obstaculo = np.ones((tam_kernel, tam_kernel), np.uint8)
        obstaculos_inflados = cv2.dilate(obstaculos, kernel_obstaculo, iterations=1)
        
        # Matriz onde é 100% fisicamente seguro pro Husky ir
        livre_seguro = cv2.bitwise_and(espaco_livre, cv2.bitwise_not(obstaculos_inflados))

        # 2. ONDE A CÂMERA AINDA NÃO VIU
        nao_visto = np.uint8(self.memoria_camera == 0) * 255

        # 3. FRONTEIRA = Lugares seguros E não vistos
        matriz_fronteiras = cv2.bitwise_and(livre_seguro, nao_visto)
        kernel_fronteira = np.ones((3, 3), np.uint8)
        bordas = cv2.morphologyEx(matriz_fronteiras, cv2.MORPH_GRADIENT, kernel_fronteira)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bordas, connectivity=8)

        fronteiras_validas = []
        rx, ry, ryaw = self.get_robot_pose()
        if rx is None:
            return None

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 5:
                # Pega um pixel real da linha para evitar cair em obstáculos (forma de 'U')
                y_idx, x_idx = np.where(labels == i)
                ponto_central_idx = len(y_idx) // 2
                cx = x_idx[ponto_central_idx]
                cy = y_idx[ponto_central_idx]
                
                ax = (cx * res) + origem_x
                ay = (cy * res) + origem_y
                dist = math.hypot(ax - rx, ay - ry)
                ang_alvo = math.atan2(ay - ry, ax - rx)
                
                # Custo pondera distância e o quanto precisa girar
                diff_ang = abs((ang_alvo - ryaw + math.pi) % (2 * math.pi) - math.pi)
                custo = dist + (diff_ang * 1.5) 
                
                fronteiras_validas.append((ax, ay, dist, custo, ang_alvo, int(cx), int(cy)))

        if not fronteiras_validas:
            self.get_logger().info("COBERTURA VISUAL 100% CONCLUÍDA!")
            return None
                
        # 4. ORDENA E TOMA A DECISÃO
        # Ordena pelo menor custo (geralmente o caminho mais suave)
        fronteiras_validas.sort(key=lambda x: x[3]) 
        alvo_x, alvo_y, dist, _, ang_alvo, px, py = fronteiras_validas[0]
        
        self.ultimo_alvo_px = (px, py)
        
        if dist > self.limite_distancia_nav:
            self.get_logger().info(f"Fronteira distante ({dist:.2f}m). Indo até o limite do mapa local...")
            # Puxa a meta para a extremidade do "local map" na mesma direção da fronteira
            alvo_x = rx + self.limite_distancia_nav * math.cos(ang_alvo)
            alvo_y = ry + self.limite_distancia_nav * math.sin(ang_alvo)
            
            # Mesmo recuando o alvo X e Y, mantemos o ang_alvo original para o robô olhar para o alvo!
        # --------------------------------------------------------------
        
        # Se o alvo está a menos de 60cm, a câmera já consegue ver apenas girando.
        if dist < 0.6:
            self.get_logger().info("Fronteira muito próxima! Apenas girando o chassi para mapear...")
            return self.criar_pose(rx, ry, ang_alvo)
        else:
            self.get_logger().info(f"Viajando para nova área: X={alvo_x:.2f}, Y={alvo_y:.2f}")
            return self.criar_pose(alvo_x, alvo_y, ang_alvo)

    def calcular_espiral(self):
        return self.criar_pose(1.5, -1.5)

    def calcular_aleatorio(self):
        return self.criar_pose(float(np.random.uniform(0.0, 3.0)), float(np.random.uniform(-3.0, 3.0)))

    def control_loop_step(self):
        if self.mapa_atual is None:
            self.get_logger().warn("⏳ Aguardando tópico de mapa...", throttle_duration_sec=3.0)
            return

        if self.voltar_base:
            self.navigator.cancelTask()
            self.navigator.goToPose(self.base_segura_pose)
            self.em_missao = True
            self.voltar_base = False
            return

        if self.mudou_algoritmo and self.conectado_rede:
            self.navigator.cancelTask()
            self.em_missao = False
            self.mudou_algoritmo = False
            return

        if self.em_missao:
            agora = self.get_clock().now()
            tempo_decorrido = (agora - self.tempo_inicio_missao).nanoseconds / 1e9
            
            if tempo_decorrido > self.timeout_missao_segundos:
                self.get_logger().warn("⚠️ Timeout! Abortando e banindo esta área...")
                self.navigator.cancelTask()
                self.em_missao = False
                
                # Se travar na parede de novo, "borda/apaga" aquele ponto da câmera
                if self.ultimo_alvo_px is not None and self.memoria_camera is not None:
                    px, py = self.ultimo_alvo_px
                    cv2.circle(self.memoria_camera, (px, py), 15, 1, -1)
                return

            if self.navigator.isTaskComplete():
                resultado = self.navigator.getResult()
                
                # Verifica se a navegação falhou ou foi cancelada pelo Nav2
                if resultado != TaskResult.SUCCEEDED:
                    self.get_logger().warn(f"❌ Nav2 falhou/abortou o alvo. Banindo esta área problemática...")
                    
                    # Borda/apaga a região alvo na memória para forçar o script a procurar outro lugar
                    if self.ultimo_alvo_px is not None and self.memoria_camera is not None:
                        px, py = self.ultimo_alvo_px
                        # Raio de 25 pixels cobre uma boa área para garantir que a fronteira bugada suma
                        cv2.circle(self.memoria_camera, (px, py), 25, 1, -1)
                else:
                    self.get_logger().info("✅ Área alcançada ou giro concluído! Buscando a próxima...")
                    
                self.em_missao = False
            return

        if not self.em_missao and self.conectado_rede:
            alvo = None
            if self.algoritmo_atual == "busca_fronteira":
                alvo = self.calcular_fronteira()
            elif self.algoritmo_atual == "busca_espiral":
                alvo = self.calcular_espiral()
            else:
                alvo = self.calcular_aleatorio()

            if alvo:
                self.navigator.goToPose(alvo)
                self.tempo_inicio_missao = self.get_clock().now()
                self.em_missao = True

def main(args=None):
    rclpy.init(args=args)
    navigator = BasicNavigator(namespace='a200_1077')
    navigator.get_logger().info("Aguardando ativação do Nav2 e SLAM...")
    navigator.waitUntilNav2Active(localizer='slam_toolbox')

    node = SearchAlgorithmNode(navigator)
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            node.control_loop_step()
            time.sleep(0.5)
    except KeyboardInterrupt:
        node.get_logger().info("Encerrando algoritmo de busca...")
    finally:
        navigator.cancelTask()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()