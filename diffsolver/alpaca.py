import json
import websocket

def on_message(ws, message):
    # 处理服务器返回的消息
    response = json.loads(message)
    print(response)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    # 发送请求
    request = {"text": "tell me what the distance between the earth and the moon"}
    ws.send(json.dumps(request))
    print("Request sent")

if __name__ == "__main__":
    # 与服务器建立WebSocket连接
    ws = websocket.WebSocketApp("ws://visiongpu58.csail.mit.edu:8080/",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.on_open = on_open
    ws.run_forever()
