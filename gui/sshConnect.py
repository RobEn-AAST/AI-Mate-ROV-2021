import paramiko
import sys

piIp = "192.168.2.2"
piUser = "pi"
piPass = "companion"

out = []
def connection():

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(piIp, piUser, piPass)

    ssh_stdin , ssh_stdout , ssh_stderr = client.exec_command('bash ROV-Gstreamer/gst0.sh')  # place your gstreamer command here

    for line in ssh_stdout:
        out.append(line.strip('\n'))

    for i in out:
        print(i.strip())

connection()
