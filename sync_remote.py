"""
script to sync tensorflow checkpoints

let script run in background while training on remote server
allows for real time updates in metrics
"""
import scp
import os
import time
import paramiko
import glob

# local_dir = sys.argv[1]
# remote_dir = sys.argv[2]
local_dir = r"../remote_logs"
remote_dir = r"/home/x2017sre/logs"

host = os.environ['CC_HOST_NAME']
user = os.environ['CC_USERNAME']
password = os.environ['CC_PASSWORD']
port = 22
def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def sync():
    os.chdir(local_dir)
    scp.get(remote_dir, recursive=True)


ssh = createSSHClient(host, port, user, password)
scp = scp.SCPClient(ssh.get_transport())
# get files
sync()
# get latest log
num_logs = len(glob.glob("../remote_logs/logs/gradient_tape/*"))
logdir = "../remote_logs/logs/gradient_tape/" + max([os.path.basename(f) for f in glob.glob("../remote_logs/logs/gradient_tape/*")])
# start tensorboard
print("serving tensorboard for {}".format(logdir))
os.system("start /B start cmd.exe @cmd /c tensorboard --logdir={}".format(logdir))

print("SCP transfer at: {} from {}".format(local_dir, remote_dir))
print("ctrl-C to end...")
while True:
    sync()
    # if new log, sync it instead of old.
    if len(glob.glob("../remote_logs/logs/gradient_tape/*")) > num_logs:
        logdir = max([os.path.basename(f) for f in glob.glob("../remote_logs/logs/gradient_tape/*")])
        print("serving tensorboard for {}".format(logdir))
        os.system("start /B start cmd.exe @cmd /c tensorboard --logdir={}".format(logdir))
        num_logs = len(glob.glob("../remote_logs/logs/gradient_tape/*"))
    time.sleep(60)
