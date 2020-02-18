EESchema Schematic File Version 4
EELAYER 30 0
EELAYER END
$Descr B 17000 11000
encoding utf-8
Sheet 1 1
Title "Mobile Cloud PCB"
Date ""
Rev "A"
Comp "SDSU S.M.I.L.E. Lab"
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L Converter_DCDC:RPM5.0-6.0 U2
U 1 1 5E3CCBF1
P 3800 3400
F 0 "U2" H 3800 3967 50  0000 C CNN
F 1 "RPM5.0-6.0" H 3800 3876 50  0000 C CNN
F 2 "Converter_DCDC:Converter_DCDC_RECOM_RPMx.x-x.0" H 3850 2600 50  0001 C CNN
F 3 "https://www.recom-power.com/pdf/Innoline/RPM-6.0.pdf" H 3775 3450 50  0001 C CNN
	1    3800 3400
	1    0    0    -1  
$EndComp
$Comp
L Converter_DCDC:RPM5.0-6.0 U3
U 1 1 5E3CD6B2
P 6700 3400
F 0 "U3" H 6700 3967 50  0000 C CNN
F 1 "RPM5.0-6.0" H 6700 3876 50  0000 C CNN
F 2 "Converter_DCDC:Converter_DCDC_RECOM_RPMx.x-x.0" H 6750 2600 50  0001 C CNN
F 3 "https://www.recom-power.com/pdf/Innoline/RPM-6.0.pdf" H 6675 3450 50  0001 C CNN
	1    6700 3400
	1    0    0    -1  
$EndComp
$Comp
L power:GNDREF #PWR01
U 1 1 5E3E001D
P 3100 2250
F 0 "#PWR01" H 3100 2000 50  0001 C CNN
F 1 "GNDREF" H 3105 2077 50  0000 C CNN
F 2 "" H 3100 2250 50  0001 C CNN
F 3 "" H 3100 2250 50  0001 C CNN
	1    3100 2250
	1    0    0    -1  
$EndComp
Wire Wire Line
	3400 3100 3150 3100
Wire Wire Line
	3150 3100 3150 2700
$Comp
L power:GNDREF #PWR03
U 1 1 5E3E1195
P 3800 3900
F 0 "#PWR03" H 3800 3650 50  0001 C CNN
F 1 "GNDREF" H 3805 3727 50  0000 C CNN
F 2 "" H 3800 3900 50  0001 C CNN
F 3 "" H 3800 3900 50  0001 C CNN
	1    3800 3900
	1    0    0    -1  
$EndComp
$Comp
L power:GNDREF #PWR07
U 1 1 5E3E1ABE
P 6700 3900
F 0 "#PWR07" H 6700 3650 50  0001 C CNN
F 1 "GNDREF" H 6705 3727 50  0000 C CNN
F 2 "" H 6700 3900 50  0001 C CNN
F 3 "" H 6700 3900 50  0001 C CNN
	1    6700 3900
	1    0    0    -1  
$EndComp
Wire Wire Line
	4850 2700 3150 2700
NoConn ~ 3400 3300
NoConn ~ 3400 3400
NoConn ~ 4200 3400
NoConn ~ 6300 3400
NoConn ~ 6300 3300
NoConn ~ 7100 3400
NoConn ~ 7100 3600
NoConn ~ 4200 3600
Wire Wire Line
	4200 3100 4450 3100
Wire Wire Line
	4450 3100 4450 3300
Wire Wire Line
	4450 3300 4200 3300
$Comp
L Device:Ferrite_Bead FB2
U 1 1 5E3F36BE
P 4700 3100
F 0 "FB2" V 4426 3100 50  0000 C CNN
F 1 "Ferrite_Bead" V 4517 3100 50  0000 C CNN
F 2 "Footprints:RESC1812X51N" V 4630 3100 50  0001 C CNN
F 3 "~" H 4700 3100 50  0001 C CNN
	1    4700 3100
	0    1    1    0   
$EndComp
Connection ~ 4450 3100
Wire Wire Line
	4550 3100 4450 3100
$Comp
L Device:C C3
U 1 1 5E3F46F6
P 5050 3250
F 0 "C3" H 5165 3296 50  0000 L CNN
F 1 "20nF" H 5165 3205 50  0000 L CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 5088 3100 50  0001 C CNN
F 3 "~" H 5050 3250 50  0001 C CNN
	1    5050 3250
	1    0    0    -1  
$EndComp
Wire Wire Line
	5050 3100 4850 3100
Wire Wire Line
	5050 3400 5050 3550
$Comp
L power:GNDREF #PWR06
U 1 1 5E3F555A
P 5050 3550
F 0 "#PWR06" H 5050 3300 50  0001 C CNN
F 1 "GNDREF" H 5055 3377 50  0000 C CNN
F 2 "" H 5050 3550 50  0001 C CNN
F 3 "" H 5050 3550 50  0001 C CNN
	1    5050 3550
	1    0    0    -1  
$EndComp
Wire Wire Line
	5050 3100 5650 3100
Wire Wire Line
	3250 4550 3250 5050
Connection ~ 5050 3100
Wire Wire Line
	7100 3100 7350 3100
Wire Wire Line
	7100 3300 7350 3300
$Comp
L Connector_Generic:Conn_01x02 J1
U 1 1 5E400429
P 2900 1950
F 0 "J1" H 2900 1750 50  0000 C CNN
F 1 " " H 2818 1716 50  0000 C CNN
F 2 "Connector_Wire:SolderWirePad_1x02_P5.08mm_Drill1.5mm" H 2900 1950 50  0001 C CNN
F 3 "~" H 2900 1950 50  0001 C CNN
	1    2900 1950
	-1   0    0    1   
$EndComp
$Comp
L Device:D D1
U 1 1 5E409987
P 4200 5100
F 0 "D1" V 4154 5179 50  0000 L CNN
F 1 "D" V 4245 5179 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 4200 5100 50  0001 C CNN
F 3 "~" H 4200 5100 50  0001 C CNN
	1    4200 5100
	0    1    1    0   
$EndComp
$Comp
L Device:D D5
U 1 1 5E40A366
P 5150 5100
F 0 "D5" V 5104 5179 50  0000 L CNN
F 1 "D" V 5195 5179 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 5150 5100 50  0001 C CNN
F 3 "~" H 5150 5100 50  0001 C CNN
	1    5150 5100
	0    1    1    0   
$EndComp
$Comp
L Device:D D2
U 1 1 5E40E46E
P 4200 5650
F 0 "D2" V 4154 5729 50  0000 L CNN
F 1 "D" V 4245 5729 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 4200 5650 50  0001 C CNN
F 3 "~" H 4200 5650 50  0001 C CNN
	1    4200 5650
	0    1    1    0   
$EndComp
$Comp
L Device:D D6
U 1 1 5E40F194
P 5150 5650
F 0 "D6" V 5104 5729 50  0000 L CNN
F 1 "D" V 5195 5729 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 5150 5650 50  0001 C CNN
F 3 "~" H 5150 5650 50  0001 C CNN
	1    5150 5650
	0    1    1    0   
$EndComp
Wire Wire Line
	4200 5500 4200 5400
Connection ~ 4200 5400
Wire Wire Line
	4200 4850 4200 4550
Wire Wire Line
	3750 5550 3950 5550
Wire Wire Line
	3950 5550 3950 5400
Wire Wire Line
	3950 5400 4200 5400
Wire Wire Line
	3750 5650 4050 5650
$Comp
L Device:D D7
U 1 1 5E41D246
P 5150 6450
F 0 "D7" V 5104 6529 50  0000 L CNN
F 1 "D" V 5195 6529 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 5150 6450 50  0001 C CNN
F 3 "~" H 5150 6450 50  0001 C CNN
	1    5150 6450
	0    1    1    0   
$EndComp
$Comp
L Device:D D3
U 1 1 5E41E18D
P 4200 6450
F 0 "D3" V 4154 6529 50  0000 L CNN
F 1 "D" V 4245 6529 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 4200 6450 50  0001 C CNN
F 3 "~" H 4200 6450 50  0001 C CNN
	1    4200 6450
	0    1    1    0   
$EndComp
$Comp
L Device:D D4
U 1 1 5E41ECF6
P 4200 7000
F 0 "D4" V 4154 7079 50  0000 L CNN
F 1 "D" V 4245 7079 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 4200 7000 50  0001 C CNN
F 3 "~" H 4200 7000 50  0001 C CNN
	1    4200 7000
	0    1    1    0   
$EndComp
$Comp
L Device:D D8
U 1 1 5E41F10C
P 5150 7000
F 0 "D8" V 5104 7079 50  0000 L CNN
F 1 "D" V 5195 7079 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 5150 7000 50  0001 C CNN
F 3 "~" H 5150 7000 50  0001 C CNN
	1    5150 7000
	0    1    1    0   
$EndComp
$Comp
L power:GNDREF #PWR04
U 1 1 5E41F5BE
P 4700 5800
F 0 "#PWR04" H 4700 5550 50  0001 C CNN
F 1 "GNDREF" H 4705 5627 50  0000 C CNN
F 2 "" H 4700 5800 50  0001 C CNN
F 3 "" H 4700 5800 50  0001 C CNN
	1    4700 5800
	1    0    0    -1  
$EndComp
Wire Wire Line
	4050 5650 4050 6100
Wire Wire Line
	4200 6300 4200 6200
Wire Wire Line
	3750 5850 3950 5850
Wire Wire Line
	3950 5850 3950 6750
Wire Wire Line
	3950 6750 4200 6750
Wire Wire Line
	4200 6750 4200 6600
Connection ~ 4200 6750
Wire Wire Line
	4200 6750 4200 6850
$Comp
L power:GNDREF #PWR05
U 1 1 5E42CEBF
P 4700 7150
F 0 "#PWR05" H 4700 6900 50  0001 C CNN
F 1 "GNDREF" H 4705 6977 50  0000 C CNN
F 2 "" H 4700 7150 50  0001 C CNN
F 3 "" H 4700 7150 50  0001 C CNN
	1    4700 7150
	1    0    0    -1  
$EndComp
Wire Wire Line
	4200 5250 4200 5400
Connection ~ 4700 5800
Wire Wire Line
	4200 5800 4700 5800
Connection ~ 4700 7150
Wire Wire Line
	4200 7150 4700 7150
Wire Wire Line
	4700 7150 5150 7150
Wire Wire Line
	5150 6600 5150 6750
Connection ~ 5150 6750
Wire Wire Line
	5150 6750 5150 6850
Wire Wire Line
	5400 6100 5400 5400
Wire Wire Line
	4050 6100 5400 6100
Wire Wire Line
	5150 6200 5150 6300
Wire Wire Line
	4700 5800 5150 5800
Wire Wire Line
	5150 5250 5150 5400
Connection ~ 5150 5400
Wire Wire Line
	5150 5400 5400 5400
Wire Wire Line
	5150 5400 5150 5500
Wire Wire Line
	5150 4850 5150 4950
Wire Wire Line
	4200 4850 5150 4850
Wire Wire Line
	4200 4950 4200 4850
Connection ~ 4200 4850
Wire Wire Line
	5400 6750 5400 7450
Wire Wire Line
	5400 7450 3850 7450
Wire Wire Line
	3850 7450 3850 5950
Wire Wire Line
	3850 5950 3750 5950
Wire Wire Line
	5150 6750 5400 6750
Connection ~ 4200 4550
Wire Wire Line
	4200 4550 5450 4550
Wire Wire Line
	3250 4550 4200 4550
NoConn ~ 2850 6450
$Comp
L power:GNDREF #PWR08
U 1 1 5E475937
P 6750 6550
F 0 "#PWR08" H 6750 6300 50  0001 C CNN
F 1 "GNDREF" H 6755 6377 50  0000 C CNN
F 2 "" H 6750 6550 50  0001 C CNN
F 3 "" H 6750 6550 50  0001 C CNN
	1    6750 6550
	1    0    0    -1  
$EndComp
$Comp
L Device:D D9
U 1 1 5E4773C6
P 7750 4700
F 0 "D9" V 7704 4779 50  0000 L CNN
F 1 "D" V 7795 4779 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 7750 4700 50  0001 C CNN
F 3 "~" H 7750 4700 50  0001 C CNN
	1    7750 4700
	0    1    1    0   
$EndComp
$Comp
L Device:D D13
U 1 1 5E4792A9
P 8700 4700
F 0 "D13" V 8654 4779 50  0000 L CNN
F 1 "D" V 8745 4779 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 8700 4700 50  0001 C CNN
F 3 "~" H 8700 4700 50  0001 C CNN
	1    8700 4700
	0    1    1    0   
$EndComp
$Comp
L Device:D D10
U 1 1 5E47976E
P 7750 5250
F 0 "D10" V 7704 5329 50  0000 L CNN
F 1 "D" V 7795 5329 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 7750 5250 50  0001 C CNN
F 3 "~" H 7750 5250 50  0001 C CNN
	1    7750 5250
	0    1    1    0   
$EndComp
$Comp
L Device:D D14
U 1 1 5E479F6C
P 8700 5250
F 0 "D14" V 8654 5329 50  0000 L CNN
F 1 "D" V 8745 5329 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 8700 5250 50  0001 C CNN
F 3 "~" H 8700 5250 50  0001 C CNN
	1    8700 5250
	0    1    1    0   
$EndComp
Wire Wire Line
	7750 5400 8250 5400
Wire Wire Line
	8700 5100 8700 5000
Wire Wire Line
	7750 4850 7750 5000
Connection ~ 7750 5000
Wire Wire Line
	7750 5000 7750 5100
Connection ~ 8700 5000
Wire Wire Line
	8700 5000 8700 4850
Wire Wire Line
	7350 5450 7550 5450
Wire Wire Line
	7550 5450 7550 5000
Wire Wire Line
	7550 5000 7750 5000
$Comp
L power:GNDREF #PWR010
U 1 1 5E48A2C3
P 8250 5400
F 0 "#PWR010" H 8250 5150 50  0001 C CNN
F 1 "GNDREF" H 8255 5227 50  0000 C CNN
F 2 "" H 8250 5400 50  0001 C CNN
F 3 "" H 8250 5400 50  0001 C CNN
	1    8250 5400
	1    0    0    -1  
$EndComp
Connection ~ 8250 5400
Wire Wire Line
	8250 5400 8700 5400
Wire Wire Line
	7350 5550 7650 5550
Wire Wire Line
	7650 5550 7650 5700
Wire Wire Line
	7650 5700 8950 5700
Wire Wire Line
	8950 5700 8950 5000
Wire Wire Line
	8950 5000 8700 5000
Wire Wire Line
	8700 4550 8700 4450
Wire Wire Line
	8700 4450 7750 4450
Wire Wire Line
	7750 4550 7750 4450
Connection ~ 7750 4450
Wire Wire Line
	5650 3100 5650 4450
$Comp
L Device:D D11
U 1 1 5E49A0DB
P 7750 6000
F 0 "D11" V 7704 6079 50  0000 L CNN
F 1 "D" V 7795 6079 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 7750 6000 50  0001 C CNN
F 3 "~" H 7750 6000 50  0001 C CNN
	1    7750 6000
	0    1    1    0   
$EndComp
$Comp
L Device:D D15
U 1 1 5E49E94D
P 8700 6000
F 0 "D15" V 8654 6079 50  0000 L CNN
F 1 "D" V 8745 6079 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 8700 6000 50  0001 C CNN
F 3 "~" H 8700 6000 50  0001 C CNN
	1    8700 6000
	0    1    1    0   
$EndComp
$Comp
L Device:D D16
U 1 1 5E49F1B4
P 8700 6550
F 0 "D16" V 8654 6629 50  0000 L CNN
F 1 "D" V 8745 6629 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 8700 6550 50  0001 C CNN
F 3 "~" H 8700 6550 50  0001 C CNN
	1    8700 6550
	0    1    1    0   
$EndComp
$Comp
L Device:D D12
U 1 1 5E49FC64
P 7750 6550
F 0 "D12" V 7704 6629 50  0000 L CNN
F 1 "D" V 7795 6629 50  0000 L CNN
F 2 "Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal" H 7750 6550 50  0001 C CNN
F 3 "~" H 7750 6550 50  0001 C CNN
	1    7750 6550
	0    1    1    0   
$EndComp
Wire Wire Line
	7750 6700 8250 6700
Wire Wire Line
	8700 6400 8700 6300
Wire Wire Line
	8700 5850 7750 5850
Wire Wire Line
	7750 6400 7750 6300
Connection ~ 7750 6300
Wire Wire Line
	7750 6300 7750 6150
Wire Wire Line
	7750 6300 7550 6300
Wire Wire Line
	7550 6300 7550 5750
Wire Wire Line
	7550 5750 7350 5750
$Comp
L power:GNDREF #PWR011
U 1 1 5E4B53A9
P 8250 6700
F 0 "#PWR011" H 8250 6450 50  0001 C CNN
F 1 "GNDREF" H 8255 6527 50  0000 C CNN
F 2 "" H 8250 6700 50  0001 C CNN
F 3 "" H 8250 6700 50  0001 C CNN
	1    8250 6700
	1    0    0    -1  
$EndComp
Connection ~ 8250 6700
Wire Wire Line
	8250 6700 8700 6700
Wire Wire Line
	7350 5850 7450 5850
Wire Wire Line
	7450 5850 7450 7000
Wire Wire Line
	7450 7000 8950 7000
Wire Wire Line
	8950 7000 8950 6300
Wire Wire Line
	8950 6300 8700 6300
Connection ~ 8700 6300
Wire Wire Line
	8700 6300 8700 6150
$Comp
L MCU_Microchip_ATmega:ATmega328P-PU U6
U 1 1 5E4EAD47
P 14300 4950
F 0 "U6" H 13950 3650 50  0000 R CNN
F 1 "ATmega328P-PU" V 13850 5250 50  0000 R CNN
F 2 "Package_DIP:DIP-28_W7.62mm" H 14300 4950 50  0001 C CIN
F 3 "http://ww1.microchip.com/downloads/en/DeviceDoc/ATmega328_P%20AVR%20MCU%20with%20picoPower%20Technology%20Data%20Sheet%2040001984A.pdf" H 14300 4950 50  0001 C CNN
	1    14300 4950
	-1   0    0    1   
$EndComp
Text Notes 12600 1400 2    197  ~ 0
XBEE3
NoConn ~ 11950 2300
NoConn ~ 11950 2400
NoConn ~ 11950 2500
NoConn ~ 11950 2600
NoConn ~ 11950 2700
NoConn ~ 11950 2800
NoConn ~ 11950 2900
NoConn ~ 11950 3000
NoConn ~ 11950 3100
NoConn ~ 11950 3200
NoConn ~ 11100 3100
NoConn ~ 11100 3000
NoConn ~ 11100 2900
NoConn ~ 11100 2800
NoConn ~ 11100 2700
NoConn ~ 11100 2600
$Comp
L power:GNDREF #PWR012
U 1 1 5E54CEAB
P 9250 2850
F 0 "#PWR012" H 9250 2600 50  0001 C CNN
F 1 "GNDREF" H 9255 2677 50  0000 C CNN
F 2 "" H 9250 2850 50  0001 C CNN
F 3 "" H 9250 2850 50  0001 C CNN
	1    9250 2850
	1    0    0    -1  
$EndComp
$Comp
L Device:C C5
U 1 1 5E55E16F
P 10200 2450
F 0 "C5" H 10315 2496 50  0000 L CNN
F 1 "10nF" H 10315 2405 50  0000 L CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 10238 2300 50  0001 C CNN
F 3 "~" H 10200 2450 50  0001 C CNN
	1    10200 2450
	1    0    0    -1  
$EndComp
Wire Wire Line
	10200 2800 10200 2600
Wire Wire Line
	11100 3200 11000 3200
Wire Wire Line
	11000 3200 11000 3350
$Comp
L power:GNDREF #PWR014
U 1 1 5E572770
P 11000 3350
F 0 "#PWR014" H 11000 3100 50  0001 C CNN
F 1 "GNDREF" H 11005 3177 50  0000 C CNN
F 2 "" H 11000 3350 50  0001 C CNN
F 3 "" H 11000 3350 50  0001 C CNN
	1    11000 3350
	1    0    0    -1  
$EndComp
Connection ~ 5650 4450
Wire Wire Line
	5650 4450 5650 4550
Wire Wire Line
	14300 3450 14300 3050
Wire Wire Line
	14300 3050 14000 3050
Wire Wire Line
	14000 3050 14000 3150
$Comp
L power:GNDREF #PWR017
U 1 1 5E57DAD3
P 14000 3150
F 0 "#PWR017" H 14000 2900 50  0001 C CNN
F 1 "GNDREF" H 14005 2977 50  0000 C CNN
F 2 "" H 14000 3150 50  0001 C CNN
F 3 "" H 14000 3150 50  0001 C CNN
	1    14000 3150
	1    0    0    -1  
$EndComp
Wire Wire Line
	11100 2400 10600 2400
$Comp
L Connector_Generic:Conn_01x02 J6
U 1 1 5E5861ED
P 9900 3300
F 0 "J6" H 9900 3100 50  0000 C CNN
F 1 " " H 9818 3066 50  0000 C CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 9900 3300 50  0001 C CNN
F 3 "~" H 9900 3300 50  0001 C CNN
	1    9900 3300
	-1   0    0    1   
$EndComp
Wire Wire Line
	10600 2400 10600 3200
Wire Wire Line
	10600 3200 10100 3200
Wire Wire Line
	10100 3300 10700 3300
Wire Wire Line
	10700 3300 10700 2500
Wire Wire Line
	10700 2500 11100 2500
Text Notes 9450 3750 0    118  ~ 0
  XBEE\nTO NANO
Text Notes 2750 2800 1    118  ~ 0
LIPO BATTERY\nTERMINALS 14.8V
Text Notes 3250 6050 1    118  ~ 0
L298HN
Text Notes 8600 1950 0    118  ~ 0
3.3V RECTIFIER
Text Notes 1900 7050 1    197  ~ 0
MOTOR DRIVERS
Text Notes 4250 4100 0    197  ~ 0
5V RECTIFIERS
Text Notes 14550 5750 1    197  ~ 0
ATMEGA328P
Wire Wire Line
	9250 2600 9250 2800
Wire Wire Line
	9550 2300 10200 2300
Connection ~ 10200 2300
Wire Wire Line
	10200 2300 11100 2300
Wire Wire Line
	10200 2800 9250 2800
Connection ~ 9250 2800
Wire Wire Line
	9250 2800 9250 2850
Wire Wire Line
	9450 7000 12300 7000
Wire Wire Line
	14300 7000 14300 6550
$Comp
L Device:R_US R1
U 1 1 5E5D0245
P 12300 4950
F 0 "R1" H 12368 4996 50  0000 L CNN
F 1 "10K" H 12368 4905 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal" V 12340 4940 50  0001 C CNN
F 3 "~" H 12300 4950 50  0001 C CNN
	1    12300 4950
	1    0    0    -1  
$EndComp
Wire Wire Line
	12300 5100 12300 7000
Wire Wire Line
	8700 4150 9450 4150
Wire Wire Line
	9450 4150 9450 7000
$Comp
L Connector_Generic:Conn_01x02 J7
U 1 1 5E5FA9EC
P 10200 4150
F 0 "J7" H 10280 4142 50  0000 L CNN
F 1 " " H 10280 4051 50  0000 L CNN
F 2 "Connector_Wire:SolderWirePad_1x02_P5.08mm_Drill1.5mm" H 10200 4150 50  0001 C CNN
F 3 "~" H 10200 4150 50  0001 C CNN
	1    10200 4150
	1    0    0    -1  
$EndComp
Wire Wire Line
	9450 4150 10000 4150
Connection ~ 9450 4150
Wire Wire Line
	10000 4250 9850 4250
Wire Wire Line
	9850 4250 9850 4400
$Comp
L power:GNDREF #PWR013
U 1 1 5E60768E
P 9850 4400
F 0 "#PWR013" H 9850 4150 50  0001 C CNN
F 1 "GNDREF" H 9855 4227 50  0000 C CNN
F 2 "" H 9850 4400 50  0001 C CNN
F 3 "" H 9850 4400 50  0001 C CNN
	1    9850 4400
	1    0    0    -1  
$EndComp
Text Notes 10400 4350 0    118  ~ 0
NANO POWER\nTERMINALS
Text Notes 9600 3250 0    59   ~ 0
Dout
Text Notes 9650 3350 0    59   ~ 0
Din
Connection ~ 12300 7000
Wire Wire Line
	12300 7000 14300 7000
$Comp
L power:GNDREF #PWR015
U 1 1 5E63BA06
P 12750 4900
F 0 "#PWR015" H 12750 4650 50  0001 C CNN
F 1 "GNDREF" H 12755 4727 50  0001 C CNN
F 2 "" H 12750 4900 50  0001 C CNN
F 3 "" H 12750 4900 50  0001 C CNN
	1    12750 4900
	1    0    0    -1  
$EndComp
$Comp
L power:GNDREF #PWR016
U 1 1 5E642E9F
P 12950 5700
F 0 "#PWR016" H 12950 5450 50  0001 C CNN
F 1 "GNDREF" H 12955 5527 50  0001 C CNN
F 2 "" H 12950 5700 50  0001 C CNN
F 3 "" H 12950 5700 50  0001 C CNN
	1    12950 5700
	1    0    0    -1  
$EndComp
$Comp
L Device:Crystal Y1
U 1 1 5E6676E1
P 12750 5300
F 0 "Y1" V 12704 5431 50  0000 L CNN
F 1 "16MHz Crystal" H 12550 5100 50  0000 L CNN
F 2 "Crystal:Crystal_HC49-U_Vertical" H 12750 5300 50  0001 C CNN
F 3 "~" H 12750 5300 50  0001 C CNN
	1    12750 5300
	0    1    1    0   
$EndComp
$Comp
L Device:C_Small C6
U 1 1 5E668E1B
P 12950 5000
F 0 "C6" H 13042 5046 50  0000 L CNN
F 1 "22pF" H 13042 4955 50  0000 L CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 12950 5000 50  0001 C CNN
F 3 "~" H 12950 5000 50  0001 C CNN
	1    12950 5000
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C7
U 1 1 5E6715C6
P 12950 5600
F 0 "C7" H 13150 5550 50  0000 R CNN
F 1 "22pF" H 13250 5650 50  0000 R CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 12950 5600 50  0001 C CNN
F 3 "~" H 12950 5600 50  0001 C CNN
	1    12950 5600
	-1   0    0    1   
$EndComp
$Comp
L Device:R_US R2
U 1 1 5E6801DA
P 13100 5300
F 0 "R2" H 13032 5254 50  0000 R CNN
F 1 "1M" H 13032 5345 50  0000 R CNN
F 2 "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal" V 13140 5290 50  0001 C CNN
F 3 "~" H 13100 5300 50  0001 C CNN
	1    13100 5300
	-1   0    0    1   
$EndComp
Connection ~ 13100 5150
Wire Wire Line
	13100 5150 13350 5150
$Comp
L Device:C_Small C8
U 1 1 5E6B8D69
P 15100 6250
F 0 "C8" H 15192 6296 50  0000 L CNN
F 1 "100nF" H 15192 6205 50  0000 L CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 15100 6250 50  0001 C CNN
F 3 "~" H 15100 6250 50  0001 C CNN
	1    15100 6250
	1    0    0    -1  
$EndComp
$Comp
L power:GNDREF #PWR018
U 1 1 5E6B93E3
P 15100 6350
F 0 "#PWR018" H 15100 6100 50  0001 C CNN
F 1 "GNDREF" H 15105 6177 50  0000 C CNN
F 2 "" H 15100 6350 50  0001 C CNN
F 3 "" H 15100 6350 50  0001 C CNN
	1    15100 6350
	1    0    0    -1  
$EndComp
Wire Wire Line
	15100 6150 14900 6150
Wire Wire Line
	14300 6550 14200 6550
Wire Wire Line
	14200 6550 14200 6450
Wire Wire Line
	13700 6050 13550 6050
Wire Wire Line
	13550 6050 13550 8300
Wire Wire Line
	13550 8300 2100 8300
Wire Wire Line
	2100 8300 2100 5850
Wire Wire Line
	2100 5450 2550 5450
Wire Wire Line
	2550 5850 2100 5850
Connection ~ 2100 5850
Wire Wire Line
	2100 5850 2100 5450
Wire Wire Line
	13700 5950 13450 5950
Wire Wire Line
	13450 5950 13450 8200
Wire Wire Line
	13450 8200 5550 8200
Wire Wire Line
	5550 8200 5550 5750
Wire Wire Line
	5550 5350 6150 5350
Wire Wire Line
	6150 5750 5550 5750
Connection ~ 5550 5750
Wire Wire Line
	5550 5750 5550 5350
Text Notes 13150 8450 0    98   ~ 0
PWM
Wire Wire Line
	13700 5850 13350 5850
Wire Wire Line
	13350 5850 13350 8100
Wire Wire Line
	13350 8100 2200 8100
Wire Wire Line
	2200 8100 2200 5650
Wire Wire Line
	2200 5250 2550 5250
Wire Wire Line
	2550 5650 2200 5650
Connection ~ 2200 5650
Wire Wire Line
	2200 5650 2200 5250
Wire Wire Line
	13700 5750 13250 5750
Wire Wire Line
	13250 5750 13250 8000
Wire Wire Line
	13250 8000 2300 8000
Wire Wire Line
	2300 8000 2300 5750
Wire Wire Line
	2300 5350 2550 5350
Wire Wire Line
	2550 5750 2300 5750
Connection ~ 2300 5750
Wire Wire Line
	2300 5750 2300 5350
Wire Wire Line
	3150 5050 3250 5050
Connection ~ 3250 5050
Wire Wire Line
	13700 6150 13700 7900
Wire Wire Line
	13700 7900 5650 7900
Wire Wire Line
	5650 7900 5650 5550
Wire Wire Line
	5650 5150 6150 5150
Wire Wire Line
	13100 5550 13100 5450
Wire Wire Line
	13100 5550 13700 5550
Connection ~ 13100 5450
Wire Wire Line
	13350 5150 13350 5450
Wire Wire Line
	13350 5450 13700 5450
Wire Wire Line
	13700 5650 13150 5650
Wire Wire Line
	13150 5650 13150 7800
Wire Wire Line
	13150 7800 5750 7800
Wire Wire Line
	5750 5250 6150 5250
Wire Wire Line
	6150 5550 5650 5550
Connection ~ 5650 5550
Wire Wire Line
	5650 5550 5650 5150
Wire Wire Line
	6150 5650 5750 5650
Wire Wire Line
	5750 5250 5750 5650
Connection ~ 5750 5650
Wire Wire Line
	5750 5650 5750 7800
$Comp
L Connector_Generic:Conn_01x02 J10
U 1 1 5E7BCEDB
P 13050 3500
F 0 "J10" H 13050 3600 50  0000 C CNN
F 1 " " H 12968 3266 50  0000 C CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 13050 3500 50  0001 C CNN
F 3 "~" H 13050 3500 50  0001 C CNN
	1    13050 3500
	0    -1   -1   0   
$EndComp
Text Notes 13009 3450 0    59   ~ 0
Rx
Text Notes 13109 3450 0    59   ~ 0
Tx
Wire Wire Line
	13700 3850 13600 3850
Wire Wire Line
	13600 3850 13600 2500
Wire Wire Line
	13700 3950 13500 3950
Wire Wire Line
	13500 3950 13500 2500
Wire Wire Line
	13700 4050 13400 4050
Wire Wire Line
	13300 4150 13700 4150
Wire Wire Line
	13150 3700 13150 4350
Wire Wire Line
	13150 4350 13700 4350
Wire Wire Line
	12750 5450 12950 5450
Wire Wire Line
	12750 5150 12950 5150
Wire Wire Line
	12950 5100 12950 5150
Connection ~ 12950 5150
Wire Wire Line
	12950 5150 13100 5150
Wire Wire Line
	12950 5450 12950 5500
Connection ~ 12950 5450
Wire Wire Line
	12950 5450 13100 5450
Wire Wire Line
	12750 4900 12950 4900
Wire Wire Line
	13050 4450 13050 3700
Wire Wire Line
	13050 4450 13700 4450
Wire Wire Line
	12300 4650 13700 4650
Wire Wire Line
	12300 4650 12300 4800
Text Notes 12900 2200 0    118  ~ 0
RESERVED FOR \nCOMMUNICATION\nWITH NANO
$Comp
L Driver_Motor:L298HN U4
U 1 1 5E3FDAAB
P 6750 5650
F 0 "U4" H 6500 6450 50  0000 C CNN
F 1 "L298HN" H 6500 6350 50  0000 C CNN
F 2 "Package_TO_SOT_THT:TO-220-15_P2.54x2.54mm_StaggerOdd_Lead4.58mm_Vertical" H 6800 5000 50  0001 L CNN
F 3 "http://www.st.com/st-web-ui/static/active/en/resource/technical/document/datasheet/CD00000240.pdf" H 6900 5900 50  0001 C CNN
	1    6750 5650
	1    0    0    -1  
$EndComp
$Comp
L Driver_Motor:L298HN U1
U 1 1 5E3FFF21
P 3150 5750
F 0 "U1" H 3150 6631 50  0000 C CNN
F 1 "L298HN" H 3150 6540 50  0000 C CNN
F 2 "Package_TO_SOT_THT:TO-220-15_P2.54x2.54mm_StaggerOdd_Lead4.58mm_Vertical" H 3200 5100 50  0001 L CNN
F 3 "http://www.st.com/st-web-ui/static/active/en/resource/technical/document/datasheet/CD00000240.pdf" H 3300 6000 50  0001 C CNN
	1    3150 5750
	1    0    0    -1  
$EndComp
Text Notes 6850 5950 1    118  ~ 0
L298HN
NoConn ~ 13700 4250
NoConn ~ 13700 5050
NoConn ~ 13700 4950
NoConn ~ 13700 5150
NoConn ~ 13700 5250
NoConn ~ 13700 4850
NoConn ~ 13700 4750
NoConn ~ 2950 6450
NoConn ~ 13700 3750
NoConn ~ 6450 6350
Wire Wire Line
	14300 6550 14300 6450
Connection ~ 14300 6550
Wire Wire Line
	7350 3300 7350 3200
Wire Wire Line
	8150 3850 8150 3700
$Comp
L power:GNDREF #PWR09
U 1 1 5E3FD0A0
P 8150 3850
F 0 "#PWR09" H 8150 3600 50  0001 C CNN
F 1 "GNDREF" H 8155 3677 50  0000 C CNN
F 2 "" H 8150 3850 50  0001 C CNN
F 3 "" H 8150 3850 50  0001 C CNN
	1    8150 3850
	1    0    0    -1  
$EndComp
Wire Wire Line
	7350 3200 7550 3200
$Comp
L Device:C C4
U 1 1 5E3FB2DD
P 8150 3550
F 0 "C4" H 8265 3596 50  0000 L CNN
F 1 "20nF" H 8265 3505 50  0000 L CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 8188 3400 50  0001 C CNN
F 3 "~" H 8150 3550 50  0001 C CNN
	1    8150 3550
	1    0    0    -1  
$EndComp
$Comp
L Device:Ferrite_Bead FB3
U 1 1 5E3F9E00
P 7700 3200
F 0 "FB3" V 7426 3200 50  0000 C CNN
F 1 "Ferrite_Bead" V 7517 3200 50  0000 C CNN
F 2 "Footprints:RESC1812X51N" V 7630 3200 50  0001 C CNN
F 3 "~" H 7700 3200 50  0001 C CNN
	1    7700 3200
	0    1    1    0   
$EndComp
Connection ~ 7350 3200
Wire Wire Line
	7350 3200 7350 3100
Wire Wire Line
	8700 3200 8700 4150
Wire Wire Line
	7850 3200 8150 3200
Wire Wire Line
	8150 3200 8150 3400
Wire Wire Line
	8150 3200 8150 2300
Wire Wire Line
	8150 2300 8950 2300
Connection ~ 8150 3200
Wire Wire Line
	8150 3200 8700 3200
$Comp
L Regulator_Linear:LD1117S33TR_SOT223 U5
U 1 1 5E5A52C3
P 9250 2300
F 0 "U5" H 9250 2542 50  0000 C CNN
F 1 "LD1117S33TR_SOT223" H 9250 2451 50  0000 C CNN
F 2 "Package_TO_SOT_SMD:SOT-223-3_TabPin2" H 9250 2500 50  0001 C CNN
F 3 "http://www.st.com/st-web-ui/static/active/en/resource/technical/document/datasheet/CD00000544.pdf" H 9350 2050 50  0001 C CNN
	1    9250 2300
	1    0    0    -1  
$EndComp
Wire Wire Line
	6300 3100 6000 3100
Wire Wire Line
	6000 3100 6000 1850
Connection ~ 3100 2150
Wire Wire Line
	3100 2150 3100 1950
Wire Wire Line
	3100 2250 3100 2150
Connection ~ 4850 1850
Wire Wire Line
	6000 1850 4850 1850
Wire Wire Line
	3650 2150 4450 2150
Connection ~ 3650 1850
Wire Wire Line
	3950 1850 3650 1850
Connection ~ 4450 1850
Wire Wire Line
	4250 1850 4450 1850
Wire Wire Line
	4450 1850 4850 1850
$Comp
L Device:Ferrite_Bead FB1
U 1 1 5E3E8DA2
P 4100 1850
F 0 "FB1" V 3826 1850 50  0000 C CNN
F 1 "Ferrite_Bead" V 3917 1850 50  0000 C CNN
F 2 "Footprints:RESC1812X51N" V 4030 1850 50  0001 C CNN
F 3 "~" H 4100 1850 50  0001 C CNN
	1    4100 1850
	0    1    1    0   
$EndComp
Connection ~ 3650 2150
$Comp
L Device:C C1
U 1 1 5E3E8170
P 3650 2000
F 0 "C1" H 3765 2046 50  0000 L CNN
F 1 "10nF" H 3765 1955 50  0000 L CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 3688 1850 50  0001 C CNN
F 3 "~" H 3650 2000 50  0001 C CNN
	1    3650 2000
	1    0    0    -1  
$EndComp
Wire Wire Line
	3100 2150 3650 2150
Wire Wire Line
	3100 1850 3650 1850
$Comp
L Device:C C2
U 1 1 5E3E532D
P 4450 2000
F 0 "C2" H 4565 2046 50  0000 L CNN
F 1 "10nF" H 4565 1955 50  0000 L CNN
F 2 "Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm" H 4488 1850 50  0001 C CNN
F 3 "~" H 4450 2000 50  0001 C CNN
	1    4450 2000
	1    0    0    -1  
$EndComp
Wire Wire Line
	4850 2700 4850 1850
NoConn ~ 6550 6350
$Comp
L power:GNDREF #PWR02
U 1 1 5E46404D
P 3150 6600
F 0 "#PWR02" H 3150 6350 50  0001 C CNN
F 1 "GNDREF" H 3155 6427 50  0000 C CNN
F 2 "" H 3150 6600 50  0001 C CNN
F 3 "" H 3150 6600 50  0001 C CNN
	1    3150 6600
	1    0    0    -1  
$EndComp
Wire Wire Line
	3150 6450 3150 6600
Wire Wire Line
	6850 4950 6850 4800
Wire Wire Line
	6750 4950 6750 4800
Wire Wire Line
	6750 4800 6850 4800
Wire Wire Line
	6750 6350 6750 6550
Wire Wire Line
	6850 4800 6850 4450
Connection ~ 6850 4800
Wire Wire Line
	5650 4450 6850 4450
Wire Wire Line
	6850 4450 7750 4450
Connection ~ 6850 4450
$Comp
L Connector_Generic:Conn_01x10 J8
U 1 1 5E4ABE30
P 11300 2700
F 0 "J8" H 11250 3200 50  0000 L CNN
F 1 " " H 11380 2601 50  0000 L CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x10_P2.00mm_Vertical" H 11300 2700 50  0001 C CNN
F 3 "~" H 11300 2700 50  0001 C CNN
	1    11300 2700
	1    0    0    -1  
$EndComp
$Comp
L Connector_Generic:Conn_01x10 J9
U 1 1 5E4C9D74
P 11750 2800
F 0 "J9" H 11700 3300 50  0000 L CNN
F 1 " " H 11830 2701 50  0000 L CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x10_P2.00mm_Vertical" H 11750 2800 50  0001 C CNN
F 3 "~" H 11750 2800 50  0001 C CNN
	1    11750 2800
	-1   0    0    1   
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J2
U 1 1 5E51627A
P 4650 5200
F 0 "J2" H 4730 5192 50  0000 L CNN
F 1 " " H 4730 5101 50  0000 L CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 4650 5200 50  0001 C CNN
F 3 "~" H 4650 5200 50  0001 C CNN
	1    4650 5200
	0    -1   -1   0   
$EndComp
Wire Wire Line
	4200 5400 4650 5400
Wire Wire Line
	4750 5400 5150 5400
$Comp
L Connector_Generic:Conn_01x02 J4
U 1 1 5E530B81
P 8200 4800
F 0 "J4" H 8280 4792 50  0000 L CNN
F 1 " " H 8280 4701 50  0000 L CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 8200 4800 50  0001 C CNN
F 3 "~" H 8200 4800 50  0001 C CNN
	1    8200 4800
	0    -1   -1   0   
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J5
U 1 1 5E53169F
P 8200 6100
F 0 "J5" H 8280 6092 50  0000 L CNN
F 1 " " H 8280 6001 50  0000 L CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 8200 6100 50  0001 C CNN
F 3 "~" H 8200 6100 50  0001 C CNN
	1    8200 6100
	0    -1   -1   0   
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J3
U 1 1 5E531F0B
P 4650 6550
F 0 "J3" H 4730 6542 50  0000 L CNN
F 1 " " H 4730 6451 50  0000 L CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 4650 6550 50  0001 C CNN
F 3 "~" H 4650 6550 50  0001 C CNN
	1    4650 6550
	0    -1   -1   0   
$EndComp
Wire Wire Line
	4200 6750 4650 6750
Wire Wire Line
	4750 6750 5150 6750
Wire Wire Line
	7750 6300 8200 6300
Wire Wire Line
	8300 6300 8700 6300
Wire Wire Line
	8300 5000 8700 5000
Wire Wire Line
	7750 5000 8200 5000
Wire Wire Line
	13400 4050 13400 2500
Wire Wire Line
	13300 4150 13300 2500
$Comp
L Connector_Generic:Conn_01x02 J11
U 1 1 5E64AF12
P 13300 2300
F 0 "J11" H 13300 2400 50  0000 C CNN
F 1 " " H 13218 2066 50  0000 C CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 13300 2300 50  0001 C CNN
F 3 "~" H 13300 2300 50  0001 C CNN
	1    13300 2300
	0    -1   -1   0   
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J12
U 1 1 5E64B2D3
P 13500 2300
F 0 "J12" H 13500 2100 50  0000 C CNN
F 1 " " H 13418 2066 50  0000 C CNN
F 2 "Connector_PinSocket_2.00mm:PinSocket_1x02_P2.00mm_Vertical" H 13500 2300 50  0001 C CNN
F 3 "~" H 13500 2300 50  0001 C CNN
	1    13500 2300
	0    -1   -1   0   
$EndComp
Wire Wire Line
	4200 6200 5150 6200
Wire Wire Line
	5150 6200 5450 6200
Wire Wire Line
	5450 6200 5450 4550
Connection ~ 5150 6200
Connection ~ 5450 4550
Wire Wire Line
	5450 4550 5650 4550
Wire Wire Line
	8700 5850 9100 5850
Wire Wire Line
	9100 5850 9100 4450
Wire Wire Line
	9100 4450 8700 4450
Connection ~ 8700 5850
Connection ~ 8700 4450
$EndSCHEMATC
