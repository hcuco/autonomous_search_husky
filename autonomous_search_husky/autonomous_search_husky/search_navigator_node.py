#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import numpy as np
import json
import cv2
import math
import threading
import time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# ros2 run autonomous_search_husky search_navigator --ros-args -p use_sim_time:=true -p timeout_missao_segundos:=20.0 -p raio_seguranca:=1.0 -r /tf:=/a200_1077/tf -r /tf_static:=/a200_1077/tf_static


class SearchNavigatorNode(Node):
    def __init__(self, navigator):
        super().__init__('search_navigator_node')

        self.declare_parameter("timeout_missao_segundos", 20.0)
        self.declare_parameter("limite_distancia_nav", 6.0)
        self.declare_parameter("raio_seguranca", 1.0)

        self.navigator = navigator
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Máquina de Estados Interna
        self.algoritmo_atual = "busca_fronteira"
        self.conectado_rede = True
        self.em_missao = False
        
        # Flags MQTT
        self.pausado = True               # Inicia pausado aguardando comando
        self.reiniciar_solicitado = False # Flag para limpar a memória
        self.mudou_algoritmo = False
        self.voltar_base = False
        self.base_segura_pose = None
        
        self.timeout_missao_segundos = self.get_parameter("timeout_missao_segundos").value
        self.limite_distancia_nav = self.get_parameter("limite_distancia_nav").value
        self.raio_seguranca = self.get_parameter("raio_seguranca").value

        self.tempo_inicio_missao = None
        self.tempo_inicio = time.time()
        
        self.mapa_atual = None
        self.memoria_camera = None

        self.pontos_banidos = []
        self.ultimo_alvo_x = None
        self.ultimo_alvo_y = None

        qos_mapa = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.create_subscription(OccupancyGrid, '/a200_1077/map', self.mapa_callback, qos_mapa)
        self.create_subscription(OccupancyGrid, '/a200_1077/mapa_cobertura_visual', self.cobertura_callback, qos_mapa)
        self.create_subscription(String, '/husky/comando_recebido', self.receber_comando, 10)
        self.create_subscription(Bool, '/status_conexao', self.atualizar_status_rede, 10)

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

    def mapa_callback(self, msg):
        self.mapa_atual = msg
        if self.base_segura_pose is None:
            self.base_segura_pose = self.criar_pose(0.0, 0.0)

    def cobertura_callback(self, msg):
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.memoria_camera = np.zeros((h, w), dtype=np.uint8)
        self.memoria_camera[grid == 0] = 1

    def atualizar_status_rede(self, msg):
        estava_conectado = self.conectado_rede
        self.conectado_rede = msg.data
        # if not self.conectado_rede and estava_conectado:
        #     self.voltar_base = True

    def receber_comando(self, msg):
        try:
            comando = json.loads(msg.data)
            acao = comando.get("acao")
            
            if acao == "iniciar_busca":
                self.algoritmo_atual = comando.get("algoritmo", "busca_fronteira")
                self.mudou_algoritmo = True
                self.pausado = False
                self.get_logger().info(f"INICIANDO BUSCA ({self.algoritmo_atual})")
                
            elif acao == "parar_busca":
                self.reiniciar_solicitado = True
                self.pausado = True
                self.get_logger().info("PARANDO BUSCA E LIMPANDO MEMÓRIA")
                
            elif acao == "pausar":
                self.pausado = True
                self.navigator.cancelTask()
                self.em_missao = False
                self.get_logger().info("BUSCA PAUSADA")
                
            elif acao == "retomar":
                self.pausado = False
                self.get_logger().info("BUSCA RETOMADA")
                
            elif acao == "voltar_base":
                self.voltar_base = True
                self.pausado = False
                self.get_logger().info("COMANDO: VOLTAR PARA A BASE")
                
            elif acao == "ir_para":
                x = comando.get("x", 0.0)
                y = comando.get("y", 0.0)
                self.pausado = True 
                self.navigator.cancelTask()
                alvo = self.criar_pose(x, y)
                self.navigator.goToPose(alvo)
                self.get_logger().info(f"OVERRIDE MANUAL: X={x}, Y={y}")
                
        except json.JSONDecodeError:
            pass

    def get_robot_pose(self):
        if self.mapa_atual is None: return None, None, None
        nome_mapa = self.mapa_atual.header.frame_id
        try:
            t = self.tf_buffer.lookup_transform(nome_mapa, 'a200_1077/base_link', rclpy.time.Time())
        except TransformException:
            try:
                t = self.tf_buffer.lookup_transform(nome_mapa, 'base_link', rclpy.time.Time())
            except TransformException:
                return None, None, None
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return x, y, yaw

    def calcular_fronteira(self):
        if self.mapa_atual is None or self.memoria_camera is None:
            return None

        if self.mapa_atual.info.width != self.memoria_camera.shape[1] or self.mapa_atual.info.height != self.memoria_camera.shape[0]:
            return None

        w, h = self.mapa_atual.info.width, self.mapa_atual.info.height
        res = self.mapa_atual.info.resolution
        origem_x = self.mapa_atual.info.origin.position.x
        origem_y = self.mapa_atual.info.origin.position.y

        grid = np.array(self.mapa_atual.data).reshape((h, w))
        raio_px = max(1, int(self.raio_seguranca / res))
        tam_kernel = raio_px * 2 + 1
        
        kernel_obstaculo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_kernel, tam_kernel))
        espaco_livre = np.uint8(grid == 0) * 255
        obstaculos = np.uint8(grid >= 50) * 255
        
        obstaculos_inflados = cv2.dilate(obstaculos, kernel_obstaculo, iterations=1)
        livre_seguro = cv2.bitwise_and(espaco_livre, cv2.bitwise_not(obstaculos_inflados))

        for (bx, by, r_px) in self.pontos_banidos:
            bx_px = int((bx - origem_x) / res)
            by_px = int((by - origem_y) / res)
            cv2.circle(livre_seguro, (bx_px, by_px), r_px, 0, -1)

        visto = np.uint8(self.memoria_camera == 1) * 255
        kernel_fronteira = np.ones((3, 3), np.uint8)
        borda_visto = cv2.morphologyEx(visto, cv2.MORPH_GRADIENT, kernel_fronteira)
        bordas = cv2.bitwise_and(borda_visto, livre_seguro)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bordas, connectivity=8)
        fronteiras_validas = []
        rx, ry, ryaw = self.get_robot_pose()
        if rx is None: return None
        
        rpx = int((rx - origem_x) / res)
        rpy = int((ry - origem_y) / res)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 5:
                y_idx, x_idx = np.where(labels == i)

                distancias_sq = (x_idx - rpx)**2 + (y_idx - rpy)**2
                melhor_idx = np.argmin(distancias_sq)
                cx, cy = x_idx[melhor_idx], y_idx[melhor_idx]
                
                ax = (cx * res) + origem_x
                ay = (cy * res) + origem_y
                dist = math.hypot(ax - rx, ay - ry)
                ang_alvo = math.atan2(ay - ry, ax - rx)
                
                diff_ang = abs((ang_alvo - ryaw + math.pi) % (2 * math.pi) - math.pi)
                custo = dist + (diff_ang * 1.5) 
                
                fronteiras_validas.append((ax, ay, dist, custo, ang_alvo))

        if not fronteiras_validas:
            duracao = time.time() - self.tempo_inicio
            self.get_logger().info(f"COBERTURA VISUAL 100% CONCLUÍDA! (Tempo: {duracao:.4f}s)")
            return None
                
        fronteiras_validas.sort(key=lambda x: x[3]) 
        alvo_x, alvo_y, dist, _, ang_alvo = fronteiras_validas[0]
        
        self.ultimo_alvo_x = alvo_x
        self.ultimo_alvo_y = alvo_y
        
        if dist > self.limite_distancia_nav:
            self.get_logger().info(f"Fronteira distante ({dist:.2f}m). Projetando alvo a {self.limite_distancia_nav}m...")
            ponto_seguro_encontrado = False
            dist_teste = self.limite_distancia_nav
            passo = res 
            
            while dist_teste > 0.4:
                tx = rx + dist_teste * math.cos(ang_alvo)
                ty = ry + dist_teste * math.sin(ang_alvo)
                tpx = int((tx - origem_x) / res)
                tpy = int((ty - origem_y) / res)
                
                if 0 <= tpx < w and 0 <= tpy < h:
                    if livre_seguro[tpy, tpx] == 255:
                        alvo_x, alvo_y = tx, ty
                        ponto_seguro_encontrado = True
                        break
                        
                dist_teste -= passo 
                
            # LÓGICA DE CONTINGÊNCIA: 9m, 8m, 7m, ..., 1m e Banimento
            if not ponto_seguro_encontrado:
                self.get_logger().warn(f"Projeção de {self.limite_distancia_nav}m falhou. Acionando contingência...")
                if dist <= 9.0:
                    alvo_x, alvo_y = self.ultimo_alvo_x, self.ultimo_alvo_y
                else:
                    distancias_contingencia = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                    contingencia_resolvida = False
                    
                    for d_cont in distancias_contingencia:
                        dist_t = d_cont
                        limite_recuo = d_cont - 0.5 
                        
                        while dist_t > limite_recuo:
                            tx = rx + dist_t * math.cos(ang_alvo)
                            ty = ry + dist_t * math.sin(ang_alvo)
                            tpx = int((tx - origem_x) / res)
                            tpy = int((ty - origem_y) / res)
                            
                            if 0 <= tpx < w and 0 <= tpy < h and livre_seguro[tpy, tpx] == 255:
                                alvo_x, alvo_y = tx, ty
                                contingencia_resolvida = True
                                break
                            dist_t -= passo
                            
                        if contingencia_resolvida:
                            break
                            
                    if not contingencia_resolvida:
                        self.get_logger().warn("Todas as contingências falharam. Banindo fronteira inalcançável!")
                        self.pontos_banidos.append((self.ultimo_alvo_x, self.ultimo_alvo_y, 30))
                        return self.criar_pose(rx, ry, ang_alvo)
                
        if dist < 0.6:
            self.get_logger().info("Fronteira muito próxima! Apenas girando o chassi...")
            return self.criar_pose(rx, ry, ang_alvo)
        else:
            self.get_logger().info(f"Viajando para área: X={alvo_x:.2f}, Y={alvo_y:.2f}")
            return self.criar_pose(alvo_x, alvo_y, ang_alvo)

    def calcular_espiral(self):
        return self.criar_pose(1.5, -1.5)

    def calcular_aleatorio(self):
        return self.criar_pose(float(np.random.uniform(0.0, 3.0)), float(np.random.uniform(-3.0, 3.0)))

    def control_loop_step(self):
        if self.mapa_atual is None or self.memoria_camera is None:
            self.get_logger().warn("Aguardando tópico de mapa e de cobertura visual...", throttle_duration_sec=3.0)
            return

        # 1. TRATA PEDIDO DE RESET/LIMPEZA (Parar Busca)
        if self.reiniciar_solicitado:
            self.get_logger().info("Limpando memória de navegação...")
            self.navigator.cancelTask()
            self.em_missao = False
            self.pontos_banidos.clear()
            self.tempo_inicio = time.time()
            self.ultimo_alvo_x = None
            self.ultimo_alvo_y = None
            self.reiniciar_solicitado = False
            return

        # 2. CONGELA O LOOP SE ESTIVER PAUSADO
        if self.pausado:
            return

        # 3. TRATA COMANDO DE RETORNAR PARA BASE
        if self.voltar_base and self.base_segura_pose:
            self.navigator.cancelTask()
            self.navigator.goToPose(self.base_segura_pose)
            self.em_missao = True
            self.voltar_base = False
            self.algoritmo_atual = "base" 
            return

        if self.mudou_algoritmo and self.conectado_rede:
            self.navigator.cancelTask()
            self.em_missao = False
            self.mudou_algoritmo = False
            return

        # 4. GERENCIA A MISSÃO EM ANDAMENTO
        if self.em_missao:
            agora = self.get_clock().now()
            tempo_decorrido = (agora - self.tempo_inicio_missao).nanoseconds / 1e9
            
            if tempo_decorrido > self.timeout_missao_segundos:
                self.get_logger().warn("Timeout! Abortando...")
                self.navigator.cancelTask()
                self.em_missao = False
                return

            if self.navigator.isTaskComplete():
                resultado = self.navigator.getResult()
                
                if self.algoritmo_atual == "base":
                    self.get_logger().info("Robô chegou na base. Entrando em modo Pausa.")
                    self.pausado = True
                    self.em_missao = False
                    return

                if resultado != TaskResult.SUCCEEDED:
                    self.get_logger().warn(f"❌ Nav2 falhou/abortou. Banindo área problemática...")
                    if self.ultimo_alvo_x is not None:
                        self.pontos_banidos.append((self.ultimo_alvo_x, self.ultimo_alvo_y, 25))
                else:
                    self.get_logger().info("✅ Área alcançada ou giro concluído! Buscando a próxima...")
                self.em_missao = False
            return

        # 5. INICIA NOVA MISSÃO DE EXPLORAÇÃO
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

    node_search = SearchNavigatorNode(navigator)
    
    executor = SingleThreadedExecutor()
    executor.add_node(node_search)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    node_search.get_logger().info("✅ Nó de Busca e Navegação iniciado. Aguardando comandos MQTT...")

    try:
        while rclpy.ok():
            node_search.control_loop_step()
            time.sleep(0.5)
    except KeyboardInterrupt:
        node_search.get_logger().info("Encerrando nó de navegação...")
    finally:
        navigator.cancelTask()
        executor.shutdown()
        node_search.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()