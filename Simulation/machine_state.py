import numpy as np
import math
import pandas as pd
from MMR_Calculator import voxel_class_numba

class MachineState:
    
    #Definition der Maschinenzustand aus Werkstoffkennwerten und maschinenpezifische Konstanten
    
    def __init__(self, k_c1, k_f1, k_p1, K_v, K_kss, x: float, y: float, z: float, tool_radius: float, tooth_amount: int, machine_coef_x: float, machine_coef_y: float,machine_coef_z: float,) -> None:
        
        self.k_c1           = k_c1          # Spez. Schnittkraft in N/mm2
        self.k_f1           = k_f1          # Spez. Vorschubkraft in N/mm2
        self.k_p1           = k_p1          # Spez. Passivkraft in N/mm2
        self.K_v            = K_v           # Verschleisskorrekturfaktor
        self.K_kss          = K_kss         # Korrekturfaktor fuer Kuehlschmierstoff
        self.x              = x             # Anstiegswert Vorschubkraft
        self.y              = y             # Anstiegswert Passivkraft
        self.z              = z             # Anstiegswert Schnittkraft
        self.tool_radius    = tool_radius
        self.tooth_amount   = tooth_amount

        #Korrekturfaroren achsenspezifisch
        self.machine_coef_x   = machine_coef_x
        self.machine_coef_y   = machine_coef_y
        self.machine_coef_z   = machine_coef_z

        #list for thooth
        self.teeth = [i * (2 * math.pi / self.tooth_amount) for i in range(self.tooth_amount)]

    def get_tool_radius(self) -> float:
        
        return self.tool_radius

    def set_tool_radius(self, new_radius) -> None:
        
        self.tool_radius = new_radius

    def set_theeth_angle(self, n: float) -> None:
        
        for i in range(len(self.teeth)):
            self.teeth[i] = (self.teeth[i] + n) % (2 * math.pi)
        
class ProcessState:

    def __init__(self, x_pos: float, y_pos: float, z_pos: float, v_x: float, v_y: float, v_z: float, v_sp: float, a_p: float, a_e: float) -> None:
        
        #Definieren Prozesszustand im Fall wenn kein Spindelgeschwindigkeit hat

        self.x_pos  = x_pos
        self.y_pos  = y_pos
        self.z_pos  = z_pos

        self.v_x    = v_x
        self.v_y    = v_y
        self.v_z    = v_z
        self.v_c    = math.sqrt(v_x**2 + v_y**2)                #v_c: Geschwindigkeit auf der x,y Ebene (Virschubgeschwindigkeit)
        self.v_ges  = math.sqrt(v_x**2 + v_y**2 + v_z**2)       #v_ges: beschreibt die gesamte Geschwindigkeit in 3D Raum

        if v_sp == 0 or self.v_c == 0:
            self.process_null_state = True
        else:
            self.process_null_state = False

        #Winkel theta beschreibt die Bewegungsrichtung der Werkzeug in x, y Ebene
        #Hier ist wichtig tan ist nur bis 90 Grad beschraenkt aber in unserem Fall geht der Winkel bis 2Pi

        if v_x > 0 and v_y >= 0:
            self.theta = math.atan(v_y/v_x)
        elif v_x < 0 and v_y >= 0:
            self.theta = math.pi + math.atan(v_y/v_x)
        elif v_x < 0 and v_y <= 0:
            self.theta = math.pi + math.atan(v_y/v_x)
        elif v_x > 0 and v_y <= 0:
            self.theta = 2 * math.pi + math.atan(v_y/v_x)
        elif v_x == 0 and v_y < 0:
            self.theta = -(math.pi/2)
        elif v_x == 0 and v_y > 0:
            self.theta = math.pi/2

        #self.theta = math.atan2(v_y, v_x)           #theta: beschreibt die Bewegungsrichtung des Werkzeugs in x,y Ebene

        def calculate_theta(v_x, v_y):
            if v_x == 0 and v_y == 0:
                self.theta = 0
                #raise ValueError("Both v_x and v_y cannot be zero.")
            self.theta = math.atan2(v_y, v_x)
            if self.theta < 0:
                self.theta += 2 * math.pi
            return self.theta
        # TODO: Prüfen
        self.theta = calculate_theta(self.v_x, self.v_y)
        

        self.v_sp   = v_sp
        self.a_e    = a_e                                       #a_e: Eingriffsbreite hier kann das Durchmesser von Werkzeug sein aufgrund von volle Zugriff
        self.a_p    = a_p                                       #a_p: Eingriffstiefe

        
    def calculate_force(self, machine_state: MachineState, frequence: int) -> tuple [float, float, float, float]:
        digits = 6
        if self.process_null_state == False:
        
            #Prozesszustand
            theta           =   self.theta
            a_p             =   self.a_p
            a_e             =   self.a_e
            v_ges           =   self.v_ges
            v_sp            =   self.v_sp * 10 / (60*60) # Umrechnung in U/s
            v_x             =   self.v_x
            v_y             =   self.v_y
            v_z             =   self.v_z

            #Maschinenzustand
            x               =   machine_state.x
            y               =   machine_state.y
            z               =   machine_state.z
            k_c1            =   machine_state.k_c1
            k_f1            =   machine_state.k_f1
            k_p1            =   machine_state.k_p1
            K_v             =   machine_state.K_v
            K_kss           =   machine_state.K_kss
            tool_radius     =   machine_state.tool_radius
            tooth_amount    =   machine_state.tooth_amount
            kappa           =   math.radians(90)
            phi_s           =   math.radians(180)                             #Schnittwinkel bei volle fraesen ist das 180 Grad
            teeth           =   machine_state.teeth

            machine_coef_x    =   machine_state.machine_coef_x
            machine_coef_y    =   machine_state.machine_coef_y
            machine_coef_z    =   machine_state.machine_coef_z

            force_rotation = np.array([[math.cos(theta), math.sin(theta), 0],
                                     [math.sin(theta), -math.cos(theta), 0],
                                     [0, 0, 1]])


            #Berechnung Vorsschub pro Zahn fz
        
            fz = v_ges / ((v_sp) * tooth_amount)
            #print(f'f_z: {fz}')

            # Berechnen Winkel pro Messung aus Spindekdrehzahl
            angle_update = v_sp / (60 * frequence) * (2*math.pi)
            machine_state.set_theeth_angle(angle_update)

            #Komponenten fuer Spannungsquerschnitt
            h = 114.6 / math.degrees(phi_s) * fz * (a_e / (tool_radius * 2) ) * math.sin(kappa)     #hier ist Ausnahme, dass unser Winkel Kappa 90 Grad ist
            b = a_p / math.sin(kappa)                  #hier auch Ausnahme, Kappa 90 Grad --> b ist gleich die Eingriefstiefe a_p

            #Spanndicke kann nicht negativ sein. Ist fuer die Kraftberechnung wichtig sonst nan Wert fuer Kraft
            if h < 0:
                h = 0

            #print(f'h: {h}')

            #Berechnung der Kraefte
            #print(h)
            F_c     = b * h ** (1-z) * k_c1 * K_v * K_kss       # Schnittkraft pro Schneide
            F_cn    = b * h ** (1-x) * k_f1 * K_v * K_kss       # Schnittnormalkraft pro Schneide #ToDo; Prüfen wie es durch teilung durch 0 kommen konnte
            F_pz    = b * h ** (1-y) * k_p1 * K_v * K_kss       # Passivkraft pro Schneide

            F_c_matrix = np.array([ [F_c], 
                                    [F_cn], 
                                    [F_pz]])
            
            F_fz    = [None] * len(teeth)
            F_fnz    = [None] * len(teeth)

            for i in range(len(teeth)):
                if teeth[i] > phi_s:

                    F_fz[i]     = 0
                    F_fnz[i]    = 0

                else:

                    F_fz[i]     = F_c * math.cos(teeth[i]) + F_cn * math.sin(teeth[i])
                    F_fnz[i]    = F_c * math.sin(teeth[i]) - F_cn * math.cos(teeth[i])


            #Zerlegung in der Kraefte in x und y Richtung

            #F_ges   = math.sqrt(F_c**2 + F_cn**2)
            #F_ges   = math.sqrt(sum(F_fz)**2 + sum(F_fnz)**2) / 2
            #print(F_ges)

            #############################################################
            #Berechnung der Kraft durch Matrix (Rotation)
            f_xyz = force_rotation @ F_c_matrix
            f_x = f_xyz[0,0] * machine_coef_x 
            f_y = f_xyz[1,0] * machine_coef_y
            f_z = f_xyz[2,0] * machine_coef_z
            #############################################################

            ###########f_x = (sum(F_fz) * math.cos(theta) + sum(F_fnz) * math.sin(theta)) / 2
            ###########f_y = (sum(F_fz) * math.sin(theta) - sum(F_fnz) * math.cos(theta)) / 2

            ##########################################################################
            #f_x     = (machine_coef_x/machine_coef_x) * math.cos(theta) * F_ges
            #print(f'f_x: {f_x}')
            #f_y     = (machine_coef_y/machine_coef_y) * math.sin(theta) * F_ges
            #print(f'f_y: {f_y}')
            ############f_z     = (machine_coef_z/machine_coef_z) * F_pz
            #print(f_z)
            f_sp    = math.sqrt(f_x**2 + f_y**2)


        else:

            f_x     = 0
            f_y     = 0
            f_z     = 0
            f_sp    = 0
            #mmr     = 0
        f_x, f_y, f_z, f_sp = round(f_x, digits), round(f_y, digits), round(f_z, digits), round(f_sp, digits)
        return f_x, f_y, f_z, f_sp