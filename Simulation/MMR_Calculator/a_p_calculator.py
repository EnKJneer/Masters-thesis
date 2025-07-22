def a_p_cal (i : int, limitation: bool, machine_state = None):

    a_p: float = 0

    limit_bool: bool = limitation

    #Limit bei Alu_Depth_Gear
    #limit_1: int = 2300
    #limit_2: int = 4800

    #Limit bei Sta_Depth_Gear
    #limit_1: int = 1500
    #limit_2: int = 2900

    #Limit bei Alu_Depth_Plate
    #limit_1: int = 3500
    #limit_2: int = 7000

    #Limit bei Sta_Depth_Plate
    #limit_1: int = 2300
    #limit_2: int = 4400

    #Limit bei Laufrad durchlauf 1
    #limit_1: int = 3200
    #limit_2: int = 5500

    #Limit bei Laufrad durchlauf 2
    #limit_1: int = 3100
    #limit_2: int = 5100

    #Limit bei Laufrad durchlauf 3
    #limit_1: int = 3000
    #limit_2: int = 5100

    #Limit bei Kühlgrill S2800
    #limit_1: int = 13350
    #limit_2: int = 26730

    #Limit bei Kühlgrill S3800
    #limit_1: int = 10100
    #limit_2: int = 20000

    #Limit bei Kühlgrill S4700
    #limit_1: int = 8300
    #limit_2: int = 16600

    #Limit bei Kühlgrill S3800 Ano
    limit_1: int = 10000
    limit_2: int = 20000

    #Limit bei Bauteil_1 Alu Durchlauf_1, 2, 3
    #limit_1: int = 26400
    #limit_2: int = 36700

    #Limit bei Bauteil_1 Stahl Durchlauf_1, 2, 3
    #limit_1: int = 15500
    #limit_2: int = 22200

    if not limit_bool:
        a_p = 6
    elif i <= limit_1:
        a_p = 3

        # Bauteil_1
        #a_p = 10
        if machine_state is not None:
            machine_state.set_tool_radius(20)
        # Code nur für Bauteil_1

    elif limit_1 < i < limit_2:
        a_p = 6

        # Bauteil_1
        #a_p = 10
        if machine_state is not None:
            machine_state.set_tool_radius(10)
        # Code nur für Bauteil_1

    elif i >= limit_2:
        a_p = 9

        # Bauteil_1
        #a_p = 5
        if machine_state is not None:
            machine_state.set_tool_radius(5)
        # Code nur für Bauteil_1

    if a_p != 0:
        return a_p
    else:
        raise ValueError('a_p = 0! Check Setting')
