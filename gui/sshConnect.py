
def connection():
    import paramiko

    piIp = "192.168.2.2"
    piUser = "pi"
    piPass = "companion"

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(piIp, piUser, piPass)

    ssh_stdin , ssh_stdout , ssh_stderr = client.exec_command('bash ROV-Gstreamer/gst0.sh & bash ROV-Gstreamer/gst1.sh')  # place your gstreamer command here


def altConnection():
    import spur

    shell = spur.SshShell(hostname="localhost", username="bob", password="password1")
    with shell:
        result = shell.run(["bash", "ROV-Gstreamer/gst0.sh", "&", "bash", "ROV-Gstreamer/gst1.sh"])