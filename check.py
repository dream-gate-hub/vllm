import requests
import subprocess
import time
from loguru import logger

def check_health(api_url):
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        return False

if __name__ == "__main__":
    cur_time = time.strftime("%Y年%m月%d日%H时%M分%S秒", time.localtime())
    logger.add("/root/running.log", mode="a")
    logger.info(f"服务重启:{cur_time}")

    # 杀死相关进程
    subprocess.run(["pkill", "-f", "/root/miniconda3/envs/vllm/bin/python3 -m vllm.entrypoints.openai.api_server"])
    time.sleep(4)

    # 启动 start.sh 脚本
    subprocess.run(["sh", "start.sh"])
    time.sleep(60)

    api_url = "http://localhost:6006"  # 替换成您的API的地址
    flag = True
    while flag:
        flag = check_health(api_url)
        # print(flag)
        time.sleep(5)  # 每隔5秒检查一次
