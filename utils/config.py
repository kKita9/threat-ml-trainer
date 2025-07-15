# Files 
ALL_FILES = {
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",  # DDoS attack
    "friday_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",  # Port scanning
    "friday_bot": "Friday-WorkingHours-Morning.pcap_ISCX.csv",  # Botnet activity
    "monday_benign": "Monday-WorkingHours.pcap_ISCX.csv",  # Pure benign traffic
    "thursday_infiltration": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",  # Infiltration
    "thursday_webattacks": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",  # Web attacks: Brute Force, XSS, SQLi
    "tuesday_bruteforce": "Tuesday-WorkingHours.pcap_ISCX.csv",  # Brute-force: FTP, SSH
    "wednesday_dos_heartbleed": "Wednesday-workingHours.pcap_ISCX.csv",  # DoS attacks + Heartbleed
}


# Attacks label from CICIDS
ATTACK_LABELS = [
    'FTP-Patator',
    'SSH-Patator',
    'DoS slowloris',
    'DoS Slowhttptest',
    'DoS Hulk',
    'DoS GoldenEye',
    'Heartbleed',
    'Web Attack – Brute Force',
    'Web Attack – XSS',
    'Web Attack – Sql Injection',
    'Infiltration',
    'Bot',
    'DDoS',
    'PortScan'
]


RELEVANT_COLUMNS = [
    'ACK Flag Count',
    'Destination Port',
    'FIN Flag Count',
    'Flow Duration',
    'PSH Flag Count',
    'SYN Flag Count',
    'Total Backward Packets',
    'Total Fwd Packets',
    'Total Length of Bwd Packets',
    'Total Length of Fwd Packets',
    'Label'
]

COMMON_CIC_SURICATA_COLUMNS = [
    'ACK Flag Count',
    'Destination IP',  # remove
    'Destination Port',
    'FIN Flag Count',
    'Flow Duration',
    'Flow ID',  # remove
    'Protocol',  # remove
    'PSH Flag Count',
    'Source IP',  # remove
    'Source Port',  # remove
    'SYN Flag Count',
    'Timestamp',  # remove
    'Total Backward Packets',
    'Total Fwd Packets',
    'Total Length of Bwd Packets',
    'Total Length of Fwd Packets'
]


cic_to_suricata_mapper = {
    'ACK Flag Count': 'tcp.ack',
    'Destination IP': 'dest_ip',
    'Destination Port': 'dest_port',
    'FIN Flag Count': 'tcp.fin',
    'Flow Duration': 'flow.start + flow.end',
    'Flow ID': 'flow_id',
    'Protocol': 'proto',
    'PSH Flag Count': 'tcp.psh',
    'Source IP': 'src_ip',
    'Source Port': 'src_port',
    'SYN Flag Count': 'tcp.syn',
    'Timestamp': 'timestamp',
    'Total Backward Packets': 'flow.pkts_toclient',
    'Total Fwd Packets': 'flow.pkts_toserver',
    'Total Length of Bwd Packets': 'flow.bytes_toclient',
    'Total Length of Fwd Packets': 'flow.bytes_toserver'
}

# unified feature names used across datasets
ELECTRIC_SCHEMA = {
    'Flow Duration': 'flow_duration',
    'Total Fwd Packets': 'fwd_pkts',
    'Total Backward Packets': 'bwd_pkts',
    'Total Length of Fwd Packets': 'fwd_bytes',
    'Total Length of Bwd Packets': 'bwd_bytes',
    'Destination Port': 'dst_port',
    'ACK Flag Count': 'ack_flag',
    'SYN Flag Count': 'syn_flag',
    'FIN Flag Count': 'fin_flag',
    'PSH Flag Count': 'psh_flag',
    'Label': 'label'
}