#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import numpy as np
import json
import cv2
import math
import threading
import time
import random
import csv
import os

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# ros2 run autonomous_search_husky search_navigator --ros-args -p use_sim_time:=true -p csv_path:="~/historico_trajetorias.csv" -p timeout_missao_segundos:=20.0 -p raio_seguranca:=2.0 -r /tf:=/a200_1077/tf -r /tf_static:=/a200_1077/tf_static


class SearchNavigatorNode(Node):
    def __init__(self, navigator):
        super().__init__('search_navigator_node')

        self.declare_parameter("timeout_missao_segundos", 20.0)
        self.declare_parameter("limite_distancia_nav", 6.0)
        self.declare_parameter("raio_seguranca", 2.0)
        self.declare_parameter("largura_faixa_zigzag", 5.0)
        self.declare_parameter("max_faixas_zigzag", 30)
        self.declare_parameter("parar_por_cobertura_visual", True)
        self.declare_parameter("csv_path", '~/pe_husky/historico_trajetorias.csv')
        
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

        self.csv_path = self.get_parameter('csv_path').value
        self.timeout_missao_segundos = self.get_parameter("timeout_missao_segundos").value
        self.limite_distancia_nav = self.get_parameter("limite_distancia_nav").value
        self.raio_seguranca = self.get_parameter("raio_seguranca").value
        self.largura_faixa_zigzag = self.get_parameter("largura_faixa_zigzag").value
        self.max_faixas_zigzag = self.get_parameter("max_faixas_zigzag").value
        self.parar_por_cobertura_visual = self.get_parameter("parar_por_cobertura_visual").value

        self.tempo_inicio_missao = None
        self.tempo_inicio = time.time()
        
        self.mapa_atual = None
        self.memoria_camera = None

        self.pontos_banidos = []
        self.ultimo_alvo_x = None
        self.ultimo_alvo_y = None

        self.exploracao_concluida = False
        self.zigzag_concluido = False

        # --- REGISTRO DE TRAJETÓRIA (RViz2 + CSV) ---
        qos_path = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Tópico único com toda a trajetória da sessão
        self.path_pub_geral = self.create_publisher(Path, '/a200_1077/trajetoria_geral', qos_path)
        self.path_geral = Path()
        self.path_geral.header.frame_id = 'map'

        # Tópicos separados por algoritmo (para colorir diferente no RViz2)
        self.pubs_path_algoritmo = {}
        self.paths_algoritmo = {}
        for alg in ['busca_fronteira', 'busca_zigzag', 'busca_espiral', 'busca_aleatorio']:
            self.pubs_path_algoritmo[alg] = self.create_publisher(
                Path, f'/a200_1077/trajetoria/{alg}', qos_path
            )
            p = Path()
            p.header.frame_id = 'map'
            self.paths_algoritmo[alg] = p

        self.ultimo_ponto_traj = None
        self.distancia_total_percorrida = 0.0

        # Arquivo CSV para análise/gráficos posteriores

        caminho_csv = os.path.expanduser(self.csv_path)
        self.arquivo_csv = open(caminho_csv, mode='w', newline='')
        self.csv_writer = csv.writer(self.arquivo_csv)
        self.csv_writer.writerow(['tempo_s', 'algoritmo', 'x', 'y', 'yaw', 'distancia_acumulada_m'])

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

    def registrar_trajetoria(self):
        rx, ry, ryaw = self.get_robot_pose()
        if rx is None:
            return

        # Só registra novo ponto se o robô andou pelo menos 15 cm (evita acúmulo parado)
        if self.ultimo_ponto_traj is not None:
            dist_passo = math.hypot(rx - self.ultimo_ponto_traj[0], ry - self.ultimo_ponto_traj[1])
            if dist_passo < 0.15:
                return
            self.distancia_total_percorrida += dist_passo

        self.ultimo_ponto_traj = (rx, ry)
        agora = self.get_clock().now().to_msg()
        pose_atual = self.criar_pose(rx, ry, ryaw)
        pose_atual.header.stamp = agora

        # 1. Atualiza e publica a trajetória geral
        self.path_geral.header.stamp = agora
        self.path_geral.poses.append(pose_atual)
        self.path_pub_geral.publish(self.path_geral)

        # 2. Atualiza e publica a trajetória específica do algoritmo ativo
        alg = getattr(self, 'algoritmo_atual', 'busca_fronteira')
        if alg in self.paths_algoritmo:
            self.paths_algoritmo[alg].header.stamp = agora
            self.paths_algoritmo[alg].poses.append(pose_atual)
            self.pubs_path_algoritmo[alg].publish(self.paths_algoritmo[alg])

        # 3. Grava no CSV para gráficos/métricas
        tempo_decorrido = time.time() - getattr(self, 'tempo_inicio', time.time())
        self.csv_writer.writerow([
            f'{tempo_decorrido:.2f}',
            alg,
            f'{rx:.3f}',
            f'{ry:.3f}',
            f'{ryaw:.3f}',
            f'{self.distancia_total_percorrida:.3f}'
        ])
        self.arquivo_csv.flush()

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
                self.exploracao_concluida = False
                self.zigzag_concluido = False
                self.pausado = False
                self.reiniciar_solicitado = True
                self.get_logger().info(f"INICIANDO BUSCA ({self.algoritmo_atual})")
                
            elif acao == "parar_busca":
                self.exploracao_concluida = False
                self.zigzag_concluido = False
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
            self.exploracao_concluida = True
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

    def calcular_zigzag(self):
        if getattr(self, 'zigzag_concluido', False):
            return None

        if self.mapa_atual is None or self.memoria_camera is None:
            return None

        if self.mapa_atual.info.width != self.memoria_camera.shape[1] or self.mapa_atual.info.height != self.memoria_camera.shape[0]:
            return None

        rx, ry, ryaw = self.get_robot_pose()
        if rx is None:
            return None

        w, h = self.mapa_atual.info.width, self.mapa_atual.info.height
        res = self.mapa_atual.info.resolution
        origem_x = self.mapa_atual.info.origin.position.x
        origem_y = self.mapa_atual.info.origin.position.y

        # 1. MÁSCARA DE SEGURANÇA
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

        y_livres, x_livres = np.where(livre_seguro == 255)
        if len(x_livres) == 0:
            self.get_logger().warn("Nenhuma área livre segura encontrada. Encerrando Zig-Zag.")
            self.zigzag_concluido = True
            return None

        min_x = (np.min(x_livres) * res) + origem_x
        max_x = (np.max(x_livres) * res) + origem_x
        min_y = (np.min(y_livres) * res) + origem_y
        max_y = (np.max(y_livres) * res) + origem_y

        passo_lateral = getattr(self, 'largura_faixa_zigzag', 3.0)
        max_faixas = getattr(self, 'max_faixas_zigzag', 30)
        tolerancia_px = max(1, int(0.35 / res))
        tol_chegada_lado = max(0.35, min(0.65, passo_lateral * 0.4))

        # 2. INICIALIZAÇÃO NO CANTO MAIS PRÓXIMO
        if not hasattr(self, 'zigzag_estado'):
            cantos = [
                (min_x, min_y,  1,  1),  # Inferior esquerdo -> varre +X, sobe +Y
                (max_x, min_y, -1,  1),  # Inferior direito  -> varre -X, sobe +Y
                (min_x, max_y,  1, -1),  # Superior esquerdo -> varre +X, desce -Y
                (max_x, max_y, -1, -1),  # Superior direito  -> varre -X, desce -Y
            ]
            canto_inicial = min(cantos, key=lambda c: math.hypot(c[0] - rx, c[1] - ry))
            _, _, self.zigzag_dir_x, self.zigzag_dir_y = canto_inicial
            self.zigzag_y_atual = canto_inicial[1]
            self.zigzag_x_parede = canto_inicial[0]
            self.zigzag_faixas_concluidas = 0
            self.zigzag_estado = "INDO_CANTO"
            self.get_logger().info(
                f"Iniciando Zig-Zag pelo canto ({canto_inicial[0]:.2f}, {canto_inicial[1]:.2f})"
            )

        # CRITÉRIO DE PARADA 1: Cobertura visual já completada (após iniciar a varredura)
        if getattr(self, 'parar_por_cobertura_visual', True) and self.zigzag_faixas_concluidas >= 1:
            nao_visto = np.uint8(self.memoria_camera == 0) * 255
            pixels_restantes = cv2.countNonZero(cv2.bitwise_and(livre_seguro, nao_visto))
            limite_ruido_px = max(10, int(0.25 / (res * res)))  # Ignora sobras < 0.25 m²
            if pixels_restantes < limite_ruido_px:
                duracao = time.time() - self.tempo_inicio
                self.get_logger().info(
                    f"ZIG-ZAG CONCLUÍDO (100% da área segura já vista pela câmera)! (Tempo: {duracao:.4f}s)"
                )
                self.zigzag_concluido = True
                return None

        alvo_x, alvo_y = None, None

        # 3. ETAPA 1: IR ATÉ O CANTO INICIAL
        if self.zigzag_estado == "INDO_CANTO":
            target_cx = min_x if self.zigzag_dir_x == 1 else max_x
            target_cy = min_y if self.zigzag_dir_y == 1 else max_y

            idx_canto = np.argmin(
                ((x_livres * res + origem_x) - target_cx) ** 2 +
                ((y_livres * res + origem_y) - target_cy) ** 2
            )
            canto_seguro_x = (x_livres[idx_canto] * res) + origem_x
            canto_seguro_y = (y_livres[idx_canto] * res) + origem_y

            if math.hypot(canto_seguro_x - rx, canto_seguro_y - ry) < 0.7:
                self.get_logger().info("Canto alcançado! Iniciando primeira faixa...")
                self.zigzag_y_atual = canto_seguro_y
                self.zigzag_estado = "VARRENDO_FAIXA"
            else:
                alvo_x, alvo_y = canto_seguro_x, canto_seguro_y

        # 4. ETAPA 2: ALTERNÂNCIA ENTRE VARREDURA HORIZONTAL (---) E PASSO LATERAL (|)
        if self.zigzag_estado in ("VARRENDO_FAIXA", "MUDANDO_FAIXA"):
            iteracoes = 0
            while iteracoes < 50:
                iteracoes += 1

                # CRITÉRIO DE PARADA 2: Limite máximo de faixas atingido
                if self.zigzag_faixas_concluidas >= max_faixas:
                    duracao = time.time() - self.tempo_inicio
                    self.get_logger().info(
                        f"ZIG-ZAG CONCLUÍDO (Limite de {max_faixas} faixas atingido)! (Tempo: {duracao:.4f}s)"
                    )
                    self.zigzag_concluido = True
                    return None

                # CRITÉRIO DE PARADA 3: Ultrapassou a parede final da quadra no eixo Y
                passou_fim_y = (
                    (self.zigzag_dir_y == 1 and self.zigzag_y_atual > max_y + 0.15) or
                    (self.zigzag_dir_y == -1 and self.zigzag_y_atual < min_y - 0.15)
                )
                if passou_fim_y:
                    duracao = time.time() - self.tempo_inicio
                    self.get_logger().info(
                        f"ZIG-ZAG 100% CONCLUÍDO ({self.zigzag_faixas_concluidas} faixas)! (Tempo: {duracao:.4f}s)"
                    )
                    self.zigzag_concluido = True
                    return None

                y_linha = max(min_y, min(max_y, self.zigzag_y_atual))
                py_alvo = int((y_linha - origem_y) / res)

                mascara_faixa = np.abs(y_livres - py_alvo) <= tolerancia_px
                x_faixa = x_livres[mascara_faixa]
                y_faixa = y_livres[mascara_faixa]

                # Linha atual bloqueada/banida -> pula para a próxima faixa
                if len(x_faixa) == 0:
                    self.zigzag_y_atual += self.zigzag_dir_y * passo_lateral
                    self.zigzag_faixas_concluidas += 1
                    if self.zigzag_estado == "VARRENDO_FAIXA":
                        self.zigzag_dir_x *= -1
                    continue

                # --- SUB-ESTADO A: MOVIMENTO RETO DE UMA PAREDE A OUTRA (-------------------) ---
                if self.zigzag_estado == "VARRENDO_FAIXA":
                    idx_extremo = np.argmax(x_faixa) if self.zigzag_dir_x == 1 else np.argmin(x_faixa)
                    fim_x = (x_faixa[idx_extremo] * res) + origem_x
                    fpx = int((fim_x - origem_x) / res)

                    if 0 <= fpx < w and 0 <= py_alvo < h and livre_seguro[py_alvo, fpx] == 255:
                        fim_y = y_linha
                    else:
                        fim_y = (y_faixa[idx_extremo] * res) + origem_y

                    # Chegou na parede oposta
                    if math.hypot(fim_x - rx, fim_y - ry) < 0.7:
                        self.zigzag_faixas_concluidas += 1
                        self.zigzag_x_parede = fim_x
                        proxima_y = self.zigzag_y_atual + (self.zigzag_dir_y * passo_lateral)
                        limite_y = max_y if self.zigzag_dir_y == 1 else min_y

                        ultrapassou = (
                            (self.zigzag_dir_y == 1 and proxima_y > max_y + 0.15) or
                            (self.zigzag_dir_y == -1 and proxima_y < min_y - 0.15)
                        )
                        if ultrapassou:
                            # Faz uma última faixa rente à parede final se restar mais de 0.8m
                            if abs(limite_y - self.zigzag_y_atual) > 0.8:
                                self.zigzag_y_atual = limite_y
                            else:
                                duracao = time.time() - self.tempo_inicio
                                self.get_logger().info(
                                    f"ZIG-ZAG 100% CONCLUÍDO ({self.zigzag_faixas_concluidas} faixas)! (Tempo: {duracao:.4f}s)"
                                )
                                self.zigzag_concluido = True
                                return None
                        else:
                            self.zigzag_y_atual = proxima_y

                        self.zigzag_estado = "MUDANDO_FAIXA"
                        self.get_logger().info(
                            f"Parede alcançada! Avançando lateralmente (|) para Y={self.zigzag_y_atual:.2f}m"
                        )
                        continue
                    else:
                        alvo_x, alvo_y = fim_x, fim_y
                        break

                # --- SUB-ESTADO B: PASSO PERPENDICULAR JUNTO À PAREDE (|) ---
                elif self.zigzag_estado == "MUDANDO_FAIXA":
                    x_candidatos = (x_faixa * res) + origem_x
                    idx_lado = np.argmin(np.abs(x_candidatos - self.zigzag_x_parede))
                    lado_x = x_candidatos[idx_lado]
                    lpx = int((lado_x - origem_x) / res)

                    if 0 <= lpx < w and 0 <= py_alvo < h and livre_seguro[py_alvo, lpx] == 255:
                        lado_y = y_linha
                    else:
                        lado_y = (y_faixa[idx_lado] * res) + origem_y

                    if math.hypot(lado_x - rx, lado_y - ry) < tol_chegada_lado:
                        self.zigzag_dir_x *= -1
                        self.zigzag_estado = "VARRENDO_FAIXA"
                        self.get_logger().info(
                            f"Passo lateral concluído! Varrendo faixa Y={lado_y:.2f}m (sentido X={self.zigzag_dir_x})"
                        )
                        continue
                    else:
                        alvo_x, alvo_y = lado_x, lado_y
                        break

            if alvo_x is None:
                self.zigzag_concluido = True
                return None

        # 5. PROJEÇÃO DE WAYPOINT INTERMEDIÁRIO SEGURO
        self.ultimo_alvo_x = alvo_x
        self.ultimo_alvo_y = alvo_y

        dist = math.hypot(alvo_x - rx, alvo_y - ry)
        ang_alvo = math.atan2(alvo_y - ry, alvo_x - rx)

        if dist > self.limite_distancia_nav:
            ponto_seguro_encontrado = False
            dist_teste = self.limite_distancia_nav
            passo = res

            while dist_teste > 0.4:
                tx = rx + dist_teste * math.cos(ang_alvo)
                ty = ry + dist_teste * math.sin(ang_alvo)
                tpx = int((tx - origem_x) / res)
                tpy = int((ty - origem_y) / res)

                if 0 <= tpx < w and 0 <= tpy < h and livre_seguro[tpy, tpx] == 255:
                    alvo_x, alvo_y = tx, ty
                    ponto_seguro_encontrado = True
                    break
                dist_teste -= passo

            if not ponto_seguro_encontrado:
                alvo_x, alvo_y = self.ultimo_alvo_x, self.ultimo_alvo_y

        self.get_logger().info(f"Zig-Zag [{self.zigzag_estado}] -> X={alvo_x:.2f}, Y={alvo_y:.2f}")
        return self.criar_pose(alvo_x, alvo_y, ang_alvo)

    def calcular_aleatorio(self):
        if self.mapa_atual is None or self.memoria_camera is None:
            return None

        if self.mapa_atual.info.width != self.memoria_camera.shape[1] or self.mapa_atual.info.height != self.memoria_camera.shape[0]:
            return None

        # --- 1. Inicializa a memória da meta global (se ainda não existir) ---
        if not hasattr(self, 'meta_global_x'):
            self.meta_global_x = None
            self.meta_global_y = None

        w, h = self.mapa_atual.info.width, self.mapa_atual.info.height
        res = self.mapa_atual.info.resolution
        origem_x = self.mapa_atual.info.origin.position.x
        origem_y = self.mapa_atual.info.origin.position.y

        grid = np.array(self.mapa_atual.data).reshape((h, w))
        
        raio_px = max(1, int(self.raio_seguranca / res))
        tam_kernel = raio_px * 2 + 1
        espaco_livre = np.uint8(grid == 0) * 255
        obstaculos = np.uint8(grid >= 50) * 255
        
        kernel_obstaculo = np.ones((tam_kernel, tam_kernel), np.uint8)
        obstaculos_inflados = cv2.dilate(obstaculos, kernel_obstaculo, iterations=1)
        livre_seguro = cv2.bitwise_and(espaco_livre, cv2.bitwise_not(obstaculos_inflados))

        nao_visto = np.uint8(self.memoria_camera == 0) * 255
        
        for (bx, by, r_px) in self.pontos_banidos:
            bx_px = int((bx - origem_x) / res)
            by_px = int((by - origem_y) / res)
            cv2.circle(nao_visto, (bx_px, by_px), r_px, 0, -1)

        rx, ry, ryaw = self.get_robot_pose()
        if rx is None: return None

        usando_meta_global = False

        # --- 2. Verifica se já temos uma Meta Global ativa e distante ---
        if self.meta_global_x is not None:
            dist_ate_global = math.hypot(self.meta_global_x - rx, self.meta_global_y - ry)
            
            if dist_ate_global < 0.6:
                # Chegou na meta (ou muito perto), limpa para sortear uma nova
                self.get_logger().info("Meta global alcançada! Buscando nova fronteira...")
                self.meta_global_x = None
                self.meta_global_y = None
            else:
                # Continua focado na meta antiga
                alvo_x = self.meta_global_x
                alvo_y = self.meta_global_y
                dist = dist_ate_global
                ang_alvo = math.atan2(alvo_y - ry, alvo_x - rx)
                usando_meta_global = True
                self.get_logger().info(f"Retomando meta global distante em X={alvo_x:.2f}, Y={alvo_y:.2f}")

        # --- 3. Só procura novas fronteiras e faz o sorteio se não tiver meta global ---
        if not usando_meta_global:
            matriz_fronteiras = cv2.bitwise_and(livre_seguro, nao_visto)
            kernel_fronteira = np.ones((3, 3), np.uint8)
            bordas = cv2.morphologyEx(matriz_fronteiras, cv2.MORPH_GRADIENT, kernel_fronteira)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bordas, connectivity=8)
            fronteiras_validas = []

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 5:
                    y_idx, x_idx = np.where(labels == i)
                    ponto_central_idx = len(y_idx) // 2
                    cx, cy = x_idx[ponto_central_idx], y_idx[ponto_central_idx]
                    
                    ax = (cx * res) + origem_x
                    ay = (cy * res) + origem_y
                    dist_fronteira = math.hypot(ax - rx, ay - ry)
                    ang_fronteira = math.atan2(ay - ry, ax - rx)
                    
                    diff_ang = abs((ang_fronteira - ryaw + math.pi) % (2 * math.pi) - math.pi)
                    custo = dist_fronteira + (diff_ang * 1.5) 
                    
                    fronteiras_validas.append((ax, ay, dist_fronteira, custo, ang_fronteira))

            if not fronteiras_validas:
                duracao = time.time() - self.tempo_inicio
                self.get_logger().info(f"COBERTURA VISUAL 100% CONCLUÍDA! (Tempo: {duracao:.4f}s)")
                self.exploracao_concluida = True
                return None

            # Sorteia uma nova e salva como a nova Meta Global
            alvo_x, alvo_y, dist, _, ang_alvo = random.choice(fronteiras_validas)
            self.meta_global_x = alvo_x
            self.meta_global_y = alvo_y

        # --- 4. Submetas (corte da distância). Funciona tanto para global quanto para nova ---
        self.ultimo_alvo_x = alvo_x
        self.ultimo_alvo_y = alvo_y
        
        if dist > self.limite_distancia_nav:
            self.get_logger().info(f"Fronteira distante ({dist:.2f}m). Buscando ponto intermediário...")
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
                
            if not ponto_seguro_encontrado:
                # IMPORTANTE: Se o caminho estiver bloqueado, precisamos resetar a meta global
                # para que ele não fique travado tentando ir pro mesmo lugar bloqueado para sempre.
                self.get_logger().warn("Reta intermediária obstruída! Cancelando meta global e girando...")
                self.meta_global_x = None
                self.meta_global_y = None
                return self.criar_pose(rx, ry, ang_alvo)
                
        if dist < 0.6:
            self.get_logger().info("Fronteira muito próxima! Apenas girando o chassi para mapear...")
            # Limpamos a meta global aqui também para garantir
            self.meta_global_x = None
            self.meta_global_y = None
            return self.criar_pose(rx, ry, ang_alvo)
        else:
            self.get_logger().info(f"Viajando para sub-meta: X={alvo_x:.2f}, Y={alvo_y:.2f}")
            return self.criar_pose(alvo_x, alvo_y, ang_alvo)

    def control_loop_step(self):
        if self.mapa_atual is None or self.memoria_camera is None:
            self.get_logger().warn("Aguardando tópico de mapa e de cobertura visual...", throttle_duration_sec=3.0)
            return

        self.registrar_trajetoria()

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
            if hasattr(self, 'zigzag_estado'):
                del self.zigzag_estado
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
        if not self.em_missao:
            rx, ry, _ = self.get_robot_pose()
            if rx is None:
                self.get_logger().warn(
                    "TF do robô indisponível no momento. Tentando novamente no próximo ciclo...",
                    throttle_duration_sec=2.0
                )
                return

            alvo = None
            if self.algoritmo_atual == "busca_fronteira":
                alvo = self.calcular_fronteira()
            elif self.algoritmo_atual == "busca_zigzag":
                alvo = self.calcular_zigzag()
            else:
                alvo = self.calcular_aleatorio()

            if alvo is not None:
                aceito = self.navigator.goToPose(alvo)
                if aceito is False:
                    self.get_logger().warn(
                        "Nav2 rejeitou ou ainda não recebeu o objetivo. Tentando novamente...",
                        throttle_duration_sec=2.0
                    )
                    return
                self.tempo_inicio_missao = self.get_clock().now()
                self.em_missao = True
            elif self.exploracao_concluida or self.zigzag_concluido:
                self.get_logger().info("Missão concluída! Aguardando novos comandos MQTT...")
                self.pausado = True
            else:
                self.get_logger().warn(
                    "Nenhum alvo gerado neste ciclo (aguardando sincronização). Tentando novamente...",
                    throttle_duration_sec=2.0
                )


def main(args=None):
    rclpy.init(args=args)
    
    navigator = BasicNavigator(namespace='a200_1077')
    navigator.get_logger().info("Aguardando ativação do Nav2 e SLAM...")
    # Por (ou use um parâmetro para alternar entre 'amcl' e 'slam_toolbox'):
    navigator.declare_parameter('localizer', 'amcl')
    localizer_name = navigator.get_parameter('localizer').value
    navigator.waitUntilNav2Active(localizer=localizer_name)


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
        if hasattr(node_search, 'arquivo_csv') and not node_search.arquivo_csv.closed:
            node_search.arquivo_csv.close()
        navigator.cancelTask()
        executor.shutdown()
        node_search.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()